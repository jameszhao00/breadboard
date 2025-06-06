/**
 * @license
 * Copyright 2024 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import type { LLMContent } from "@breadboard-ai/types";
import {
  isFunctionCallCapabilityPart,
  isFunctionResponseCapabilityPart,
  isInlineData,
  isStoredData,
  isTextCapabilityPart,
} from "@google-labs/breadboard";
import { LitElement, TemplateResult, css, html, nothing } from "lit";
import { customElement, property } from "lit/decorators.js";
import { cache } from "lit/directives/cache.js";
import { classMap } from "lit/directives/class-map.js";
import { map } from "lit/directives/map.js";
import { until } from "lit/directives/until.js";
import { markdown } from "../../../directives/markdown.js";
import { ToastEvent, ToastType } from "../../../events/events.js";
import { appendToDocUsingDriveKit } from "../../google-drive/append-to-doc-using-drive-kit.js";
import { tokenVendorContext } from "../../elements.js";
import { consume } from "@lit/context";
import type { TokenVendor } from "@breadboard-ai/connection-client";
import "./export-toolbar.js";
import { styleMap } from "lit/directives/style-map.js";

const PCM_AUDIO = "audio/l16;codec=pcm;rate=24000";

const documentStyles = getComputedStyle(document.documentElement);

type ValidColorStrings = `#${string}` | `--${string}`;

export function getGlobalColor(
  name: ValidColorStrings,
  defaultValue: ValidColorStrings = "#333333"
) {
  const value = documentStyles.getPropertyValue(name)?.replace(/^#/, "");
  const valueAsNumber = parseInt(value || defaultValue, 16);
  if (Number.isNaN(valueAsNumber)) {
    return `#${(0xff00ff).toString(16)}`;
  }
  return `#${valueAsNumber.toString(16).padStart(6, "0")}`;
}

@customElement("bb-llm-output")
export class LLMOutput extends LitElement {
  @property()
  accessor value: LLMContent | null = null;

  @property({ reflect: true })
  accessor clamped = true;

  @property({ reflect: true })
  accessor lite = false;

  @property({ reflect: true })
  accessor showExportControls = false;

  @consume({ context: tokenVendorContext })
  accessor tokenVendor!: TokenVendor;

  #partDataURLs = new Map<number, string>();

  static styles = css`
    :host {
      display: block;
      position: relative;
      border: var(--output-border-width, 2px) solid
        var(--output-border-color, var(--bb-neutral-300));
      border-radius: var(--output-border-radius, var(--bb-grid-size));
      margin-bottom: var(--bb-grid-size-2);
    }

    :host([clamped="true"]) {
      resize: vertical;
      overflow: auto;
      height: 200px;
      min-height: var(--bb-grid-size-6);
    }

    :host([lite="true"]) {
      border: 1px solid var(--output-lite-border-color, var(--bb-neutral-100));
      background: var(--output-lite-background-color, var(--bb-neutral-0));
    }

    .content {
      display: block;
      position: relative;
      margin-bottom: var(--bb-grid-size-2);
      padding: var(--output-padding, var(--bb-grid-size-3)) 0;
      overflow-y: auto;
      max-height: var(--bb-llm-output-content-max-height, unset);
    }

    .content:last-of-type {
      margin-bottom: 0;
    }

    .value {
      display: flex;
      flex-direction: column;
      position: relative;

      margin: 0 var(--output-value-margin-x, var(--bb-grid-size-3));
      font: normal var(--bb-body-medium) / var(--bb-body-line-height-medium)
        var(--bb-font-family);
      color: var(--bb-neutral-900);

      padding: 0 var(--output-value-padding-x, var(--bb-grid-size-3));

      white-space: normal;
      border-radius: initial;
      user-select: text;

      &:has(> img),
      &:has(> .copy-image-to-clipboard),
      &:has(> video),
      &:has(> audio) {
        justify-content: center;
        align-items: center;
        padding: var(--bb-grid-size-2) 0;
      }

      & pre {
        font: normal var(--bb-body-small) / var(--bb-body-line-height-small)
          var(--bb-font-family);
      }
    }

    .value img,
    .value video,
    .value audio {
      width: 100%;
      max-width: 360px;
    }

    .value img,
    .value video,
    iframe.html-view {
      outline: 1px solid var(--bb-neutral-300);
      border-radius: var(--bb-grid-size);
    }

    iframe.html-view {
      border: none;
      width: 100%;
      height: 600px;
    }

    .value .plain-text {
      white-space: pre;
      font: normal var(--bb-body-small) / var(--bb-body-line-height-small)
        var(--bb-font-family-mono);
      color: var(--bb-neutral-600);
    }

    .value.markdown {
      line-height: 1.5;
      overflow-x: auto;

      & a {
        color: var(--bb-ui-700);
      }
    }

    .value * {
      margin: var(--bb-grid-size) 0;
    }

    .value h1 {
      font: 500 var(--bb-title-large) / var(--bb-title-line-height-large)
        var(--bb-font-family);

      margin: var(--bb-grid-size-6) 0 var(--bb-grid-size-2) 0;
    }

    .value h2 {
      font: 500 var(--bb-title-medium) / var(--bb-title-line-height-medium)
        var(--bb-font-family);

      margin: var(--bb-grid-size-4) 0 var(--bb-grid-size-2) 0;
    }

    .value h3,
    .value h4,
    .value h5 {
      font: 500 var(--bb-title-small) / var(--bb-title-line-height-small)
        var(--bb-font-family);

      margin: var(--bb-grid-size-3) 0 var(--bb-grid-size-2) 0;
    }

    .value p {
      font: 400 var(--bb-body-medium) / var(--bb-body-line-height-medium)
        var(--bb-font-family);

      margin: 0 0 var(--bb-grid-size-2) 0;
      white-space: pre-line;

      & strong:only-child {
        margin: var(--bb-grid-size-2) 0 0 0;
      }
    }

    .value h1:first-of-type,
    .value h2:first-of-type,
    .value h3:first-of-type,
    .value h4:first-of-type,
    .value h5:first-of-type {
      margin-top: 0;
    }

    .value p:last-of-type {
      margin-bottom: 0;
    }

    .value.no-data {
      font: normal var(--bb-body-small) / var(--bb-body-line-height-small)
        var(--bb-font-family-mono);
    }

    :host([lite="true"]) .value {
      margin: 0;
    }

    .play-audio {
      background: var(--bb-neutral-0) var(--bb-icon-sound) 6px 3px / 16px 16px
        no-repeat;
      border-radius: 20px;
      color: var(--bb-neutral-900);
      border: 1px solid var(--bb-neutral-600);
      height: 24px;
      padding: 0 16px 0 28px;
      cursor: pointer;
      opacity: 0.5;
    }

    .play-audio:hover,
    .play-audio:focus {
      opacity: 1;
    }

    .copy-image-to-clipboard {
      position: relative;

      & button {
        width: 32px;
        height: 32px;
        background: var(--bb-neutral-0) var(--bb-icon-copy-to-clipboard) center
          center / 20px 20px no-repeat;
        position: absolute;
        top: 50%;
        left: 50%;
        translate: -50% -50%;
        border-radius: 50%;
        cursor: pointer;
        border: 1px solid var(--bb-neutral-300);
        font-size: 0;
        opacity: 0;
        transition: opacity 0.2s cubic-bezier(0, 0, 0.3, 1);
      }

      &:hover button {
        opacity: 1;
      }
    }

    bb-export-toolbar {
      position: absolute;
      top: -16px;
      right: var(--export-x, 16px);
      z-index: 1;
    }
  `;

  connectedCallback(): void {
    super.connectedCallback();
    this.#clearPartDataURLs();
  }

  disconnectedCallback(): void {
    super.disconnectedCallback();
    this.#clearPartDataURLs();
  }

  #clearPartDataURLs() {
    for (const url of this.#partDataURLs.values()) {
      URL.revokeObjectURL(url);
    }

    this.#partDataURLs.clear();
  }

  // TODO: Store this value rather than converting it on the fly.
  async #convertToAudioBuffer(url: string) {
    const TO_FLOAT = 32768;
    const sampleRate = 24000;
    const audio = await fetch(url);
    const arrayBuffer = await audio.arrayBuffer();
    const audioDataIn = new Int16Array(arrayBuffer);
    const audioDataOut = new Float32Array(audioDataIn.length);

    const duration = audioDataOut.length / sampleRate;
    const audioCtx = new OfflineAudioContext({
      length: audioDataOut.length,
      sampleRate,
    });

    const audioBuffer = audioCtx.createBuffer(
      1,
      sampleRate * duration,
      sampleRate
    );

    for (let i = 0; i < audioDataIn.length; i++) {
      audioDataOut[i] = audioDataIn[i] / TO_FLOAT;
    }

    audioBuffer.copyToChannel(audioDataOut, 0);
    const source = audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioCtx.destination);
    source.start();

    return audioCtx.startRendering();
  }

  render() {
    const canCopy = this.showExportControls && "ClipboardItem" in window;

    return this.value && this.value.parts.length
      ? html` ${this.showExportControls
          ? html`<bb-export-toolbar .value=${this.value}></bb-export-toolbar>`
          : nothing}
        ${map(this.value.parts, (part, idx) => {
          let value: TemplateResult | symbol = nothing;
          if (isTextCapabilityPart(part)) {
            if (part.text === "") {
              value = html`${markdown("[Continue]")}`;
            } else {
              value = html`${markdown(part.text)}`;
            }
          } else if (isInlineData(part)) {
            const key = idx;
            let partDataURL: Promise<string> = Promise.resolve("No source");
            if (this.#partDataURLs.has(key)) {
              partDataURL = Promise.resolve(this.#partDataURLs.get(key)!);
            } else if (
              part.inlineData.data !== "" &&
              !part.inlineData.mimeType.startsWith("text")
            ) {
              const dataURL = `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;
              partDataURL = fetch(dataURL)
                .then((response) => response.blob())
                .then((data) => {
                  const url = URL.createObjectURL(data);
                  this.#partDataURLs.set(key, url);
                  return url;
                });
            }
            const tmpl = partDataURL.then((url: string) => {
              if (part.inlineData.mimeType.startsWith("image")) {
                return cache(html`
                  ${canCopy
                    ? html` <div class="copy-image-to-clipboard">
                        <img src="${url}" alt="LLM Image" />
                        <button
                          @click=${async () => {
                            const data = await fetch(url);
                            const imageData = await data.blob();

                            await navigator.clipboard.write([
                              new ClipboardItem({
                                [part.inlineData.mimeType]: imageData,
                              }),
                            ]);

                            this.dispatchEvent(
                              new ToastEvent(
                                "Copied image to Clipboard",
                                ToastType.INFORMATION
                              )
                            );
                          }}
                        >
                          Copy image to clipboard
                        </button>
                      </div>`
                    : html`<img src="${url}" alt="LLM Image" />`}
                `);
              }
              if (part.inlineData.mimeType.startsWith("audio")) {
                const audioHandler = fetch(url)
                  .then((r) => r.blob())
                  .then((data) => {
                    if (
                      part.inlineData.mimeType.toLocaleLowerCase() === PCM_AUDIO
                    ) {
                      return new Blob([data], { type: PCM_AUDIO });
                    }

                    return data;
                  })
                  .then((audioFile) => {
                    const colorLight =
                      this.value?.role === "model"
                        ? getGlobalColor("--bb-generative-400")
                        : getGlobalColor("--bb-ui-400");
                    const colorMid =
                      this.value?.role === "model"
                        ? getGlobalColor("--bb-generative-500")
                        : getGlobalColor("--bb-ui-500");
                    const colorDark =
                      this.value?.role === "model"
                        ? getGlobalColor("--bb-generative-600")
                        : getGlobalColor("--bb-ui-600");

                    console.log(colorLight, colorMid, colorDark);

                    return cache(
                      html`<div class="play-audio-container">
                        <bb-audio-handler
                          .audioFile=${audioFile}
                          .color=${colorLight}
                          style=${styleMap({
                            "--color-light": colorLight,
                            "--color-mid": colorMid,
                            "--color-dark": colorDark,
                          })}
                        ></bb-audio-handler>
                      </div>`
                    );
                  });

                return cache(html`${until(audioHandler)}`);
              }
              if (part.inlineData.mimeType.startsWith("video")) {
                return cache(html`<video src="${url}" controls />`);
              }
              if (part.inlineData.mimeType.startsWith("text")) {
                return cache(
                  // prettier-ignore
                  html`<div class="plain-text">${atob(part.inlineData.data)}</div>`
                );
              }
            });
            value = html`${until(tmpl)}`;
          } else if (
            isFunctionCallCapabilityPart(part) ||
            isFunctionResponseCapabilityPart(part)
          ) {
            value = html` <bb-json-tree .json=${part}></bb-json-tree>`;
          } else if (isStoredData(part)) {
            const url = part.storedData.handle;
            if (!url) {
              value = html`<div>Failed to retrieve stored data</div>`;
            } else {
              const { mimeType } = part.storedData;
              const getData = async () => {
                const response = await fetch(url);
                return response.text();
              };
              if (mimeType.startsWith("image")) {
                value = html`<img src="${url}" alt="LLM Image" />`;
              }
              if (mimeType.startsWith("audio")) {
                value = html`<audio src="${url}" controls />`;
              }
              if (mimeType.startsWith("video")) {
                value = html`<video src="${url}" controls />`;
              }
              if (mimeType.startsWith("text")) {
                value = html`<div class="plain-text">${until(getData())}</div>`;
              }
            }
          } else {
            value = html`Unrecognized part`;
          }
          return html`<div class="content">
            <span
              class=${classMap({
                value: true,
                markdown: isTextCapabilityPart(part),
              })}
              >${value}</span
            >
          </div>`;
        })}`
      : html`<span class="value no-data">No data set</span>`;
  }

  async #onClickSaveToGoogleDriveButton() {
    if (!this.value) {
      console.error("Error saving to Google Drive: No value");
      return;
    }
    if (!this.tokenVendor) {
      console.error("Error saving to Google Drive: No token vendor");
      return;
    }
    const { url } = await appendToDocUsingDriveKit(
      this.value,
      `Breadboard Demo (${new Date().toLocaleDateString("en-US")})`,
      this.tokenVendor
    );
    this.dispatchEvent(
      new ToastEvent(
        // HACK: Toast messages are typed to only allow strings, but actually
        // they just directly render the value, so a TemplateResult works too,
        // letting us embed a link.
        html`Content saved to
          <a href=${url} target="_blank">Google Doc</a>` as unknown as string,
        ToastType.INFORMATION
      )
    );
  }
}
