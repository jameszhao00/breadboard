/**
 * @fileoverview Manages Gemini prompt.
 */

import invokeBoard from "@invoke";

import gemini, {
  type GeminiInputs,
  type GeminiOutputs,
  type Candidate,
} from "./gemini";
import { ToolManager } from "./tool-manager";
import { ok, err, toLLMContent, addUserTurn } from "./utils";

export { GeminiPrompt };

type FunctionResponsePart = {
  functionResponse: {
    name: string;
    response: object;
  };
};

function textToJson(content: LLMContent): LLMContent {
  return {
    ...content,
    parts: content.parts.map((part) => {
      if ("text" in part) {
        try {
          return { json: JSON.parse(part.text) };
        } catch (e) {
          // Just return the original part if it's not valid JSON.
        }
      }
      return part;
    }),
  };
}

function mergeLastParts(contexts: LLMContent[][]): LLMContent {
  const parts: DataPart[] = [];
  for (const context of contexts) {
    const last = context.at(-1);
    if (!last) continue;
    if (!last.parts) continue;
    parts.push(...last.parts);
  }
  return {
    parts,
    role: "user",
  };
}

export type ValidatorFunction = (response: LLMContent) => Outcome<void>;

export type GeminiPromptOutput = {
  last: LLMContent;
  all: LLMContent[];
  candidate: Candidate;
};

export type GeminiPromptInvokeOptions = GeminiPromptOptions;

export type GeminiPromptOptions = {
  allowToolErrors?: boolean;
  validator?: ValidatorFunction;
  toolManager?: ToolManager;
};

const MAX_SEQUENTIAL_TOOL_CALLS = 15;

class GeminiPrompt {
  readonly options: GeminiPromptOptions;

  calledTools: boolean = false;
  // Useful for detecting if subgraphs were invoked, based on whether a tool was invoked with
  // 'passContext' set to true.
  calledCustomTools: boolean = false;

  constructor(
    public readonly inputs: GeminiInputs,
    options?: ToolManager | GeminiPromptOptions
  ) {
    this.options = this.#reconcileOptions(options);
  }

  #reconcileOptions(
    options?: ToolManager | GeminiPromptOptions
  ): GeminiPromptOptions {
    if (!options) return {};
    if (options instanceof ToolManager) {
      return { toolManager: options };
    }
    return options;
  }

  #normalizeArgs(a: object, passContext?: boolean) {
    if (!passContext) return a;
    const args = a as Record<string, unknown>;
    const context = [...this.inputs.body.contents];
    const hasContext = "context" in args;
    if (hasContext) {
      const argContext = args.context as LLMContent[];
      context.push(...argContext);
    }
    return { ...args, context };
  }

  async invoke(): Promise<Outcome<GeminiPromptOutput>> {
    this.calledTools = false;
    this.calledCustomTools = false;
    const { allowToolErrors, validator, toolManager } = this.options;

    let currentApiContents: LLMContent[] = [...this.inputs.body.contents];
    let lastCandidate: Candidate | undefined;
    let finalModelResponseContent: LLMContent | undefined;
    let loopIteration = 0;

    for (
      loopIteration = 0;
      loopIteration < MAX_SEQUENTIAL_TOOL_CALLS;
      loopIteration++
    ) {
      const currentGeminiInputs: GeminiInputs = {
        ...this.inputs,
        body: {
          ...this.inputs.body,
          contents: currentApiContents,
        },
      };

      // console.log("Calling Gemini API, iteration:", loopIteration + 1, "with contents:", JSON.stringify(currentApiContents, null, 2));
      const invoking = await gemini(currentGeminiInputs);

      if (!ok(invoking)) {
        console.error("Error from Gemini API:", invoking.$error);
        return invoking;
      }
      if ("context" in invoking) {
        return err(
          "Invalid output from Gemini -- expected candidates, got context"
        );
      }

      const candidate = invoking.candidates.at(0);
      const modelResponsePart = candidate?.content;
      lastCandidate = candidate;

      if (!modelResponsePart) {
        return err(
          `No content from Gemini. Finish reason: ${candidate?.finishReason}`
        );
      }
      if (!modelResponsePart.parts || modelResponsePart.parts.length === 0) {
        if (candidate?.finishReason === "STOP") {
          finalModelResponseContent = modelResponsePart;
          // console.log("Gemini returned empty parts with STOP reason.");
          break;
        }
        return err(
          `Gemini failed to generate result parts. Finish reason: ${candidate?.finishReason}`
        );
      }

      currentApiContents = [...currentApiContents, modelResponsePart];
      finalModelResponseContent = modelResponsePart;

      const hasFunctionCall = modelResponsePart.parts.some(
        (part) => "functionCall" in part
      );

      if (hasFunctionCall && toolManager) {
        // console.log("Gemini response includes function call, processing with ToolManager.");
        this.calledTools = true;
        const turnToolResultsLLMContent: LLMContent[][] = [];
        const turnToolErrors: string[] = [];

        await toolManager.processResponse(
          modelResponsePart,
          async ($board, args, passContext, functionName) => {
            // console.log("Executing tool via ToolManager:", functionName, "with args:", args);
            this.calledTools = true;
            if (passContext) {
              this.calledCustomTools = true;
            }
            const callingTool = await invokeBoard({
              $board,
              ...this.#normalizeArgs(args, passContext),
            });

            if ("$error" in callingTool) {
              turnToolErrors.push(JSON.stringify(callingTool.$error));
            } else if (functionName === undefined) {
              turnToolErrors.push(
                `No function name for tool response ${JSON.stringify(
                  callingTool
                )}`
              );
            } else {
              let responsePartData: FunctionResponsePart;
              if (passContext) {
                if (!("context" in callingTool)) {
                  turnToolErrors.push(
                    `No "context" port in outputs of subgraph "${$board}"`
                  );
                  return;
                }
                const response = {
                  ["value"]: JSON.stringify(
                    callingTool.context as LLMContent[]
                  ),
                };
                responsePartData = {
                  functionResponse: { name: functionName, response },
                };
              } else {
                responsePartData = {
                  functionResponse: {
                    name: functionName,
                    response: callingTool,
                  },
                };
              }
              const toolResponseContent: LLMContent = {
                role: "tool",
                parts: [responsePartData],
              };
              // console.log("Tool execution result for", functionName, ":", toolResponseContent);
              turnToolResultsLLMContent.push([toolResponseContent]);
            }
          }
        );

        if (turnToolErrors.length && !allowToolErrors) {
          return err(
            `Calling tools generated errors: ${turnToolErrors.join(",")}`
          );
        }
        if (turnToolErrors.length && allowToolErrors) {
          console.warn("Tool errors occurred but are allowed:", turnToolErrors);
          if (turnToolResultsLLMContent.length === 0) {
            // console.log("Breaking due to tool errors (allowed) but no successful tool results.");
            break;
          }
        }

        if (turnToolResultsLLMContent.length > 0) {
          const mergedToolResponses = mergeLastParts(turnToolResultsLLMContent);
          currentApiContents = [...currentApiContents, mergedToolResponses];
          // console.log("Added tool responses to API contents for next iteration.");
        } else {
          // console.log("No tool results to send back, breaking sequence.");
          break;
        }
      } else {
        // console.log("No function call in Gemini response or no ToolManager, sequence ends.");
        if (validator) {
          const validating = validator(modelResponsePart);
          if (!ok(validating)) {
            console.error(
              "Validation error on final response:",
              validating.$error
            );
            return validating;
          }
        }
        break;
      }
    }

    if (loopIteration >= MAX_SEQUENTIAL_TOOL_CALLS) {
      return err(
        `Maximum sequential tool call limit (${MAX_SEQUENTIAL_TOOL_CALLS}) reached.`
      );
    }

    if (!finalModelResponseContent) {
      return err("No final model response content obtained after loop.");
    }

    // console.log("gemini-prompt final model content:", finalModelResponseContent);
    const outputParts =
      this.inputs.body.generationConfig?.responseMimeType === "application/json"
        ? [textToJson(finalModelResponseContent)]
        : [finalModelResponseContent];

    return {
      all: outputParts,
      last: outputParts.at(-1)!,
      candidate: lastCandidate!,
    };
  }
}
