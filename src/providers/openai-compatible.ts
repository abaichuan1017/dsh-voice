/**
 * Transcription provider for any endpoint speaking OpenAI's `/v1/audio/transcriptions` request and
 * response. One shape reaches hosted APIs and self-hosted servers alike, so "use a cloud model" and
 * "run Whisper on this machine behind a local server" are the same provider with a different
 * `baseUrl`.
 *
 * The API key is addressed by reference, never stored: `apiKeyEnv` names an environment variable and
 * the value is resolved from `ctx.credentials` at the start of every call, so a rotated key reaches
 * the next request with no restart. Omitting `apiKeyEnv` targets a local server that wants no
 * authorization at all.
 * @module @deepseek-ai/dsh-transcription-openai-compatible
 */

import { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { credentialRef } from '@deepseek-ai/dsh-credentials'
import type { CredentialRef } from '@deepseek-ai/dsh-credentials/types'
import {
  TranscriptionEngine,
  TranscriptionError,
  type AudioClip,
  type TranscriptResult,
  type TranscriptionProviderInfo,
} from '../transcription/index.ts'
import {
  ACCEPTED_MEDIA_TYPES,
  bareMediaType,
  classifyHttpFailure,
  mediaTypeExtension,
  parseTranscriptionBody,
} from './openai-protocol.ts'

/** Provider identity reported by `describe()`; equal to this package's plugin name. */
const PROVIDER_NAME = 'transcription-openai-compatible'

/** Deployment configuration for one OpenAI-compatible transcription endpoint. */
export interface Config {
  /**
   * Origin and path prefix of the endpoint, without the `/audio/transcriptions` suffix — for
   * example `https://api.openai.com/v1` or `http://127.0.0.1:8000/v1`. A trailing slash is accepted
   * and normalized away.
   */
  baseUrl: string
  /** Transcription model the endpoint should use, such as `whisper-1` or `Systran/faster-whisper-small`. */
  model: string
  /**
   * Environment-variable name holding the bearer token. Omit it for a local server that requires no
   * authorization; a named-but-empty variable reads as unconfigured rather than as an empty key.
   */
  apiKeyEnv?: string
  /**
   * Deadline for one request, measured from dispatch to response headers plus body. Transcription
   * cost scales with clip length, so this is a deployment choice rather than a fixed constant.
   */
  timeoutMs: number
}

/**
 * Strip one trailing slash so `${baseUrl}/audio/transcriptions` never doubles it.
 * @param baseUrl - the configured endpoint prefix.
 * @returns the prefix without a trailing slash.
 */
function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl
}

/**
 * Read the endpoint's error body without letting a second failure mask the first.
 * @param response - the non-OK response.
 * @param signal - caller-owned cancellation, preserved even while reading an error response.
 * @param timeout - the deadline covering response headers and body.
 * @returns a bounded diagnostic string, empty when the body cannot be read.
 */
async function readErrorBody(response: Response, signal: AbortSignal, timeout: AbortSignal): Promise<string> {
  try {
    return (await response.text()).slice(0, 512)
  } catch (error) {
    if (signal.aborted || timeout.aborted) throwTransportFailure(error, signal, timeout.aborted)
    // A body that cannot be read adds nothing to the status the caller already has, and letting
    // this throw would replace a classified provider-rejected failure with a stream error.
    return ''
  }
}

/**
 * Classify a transport-level failure. A caller-initiated abort is rethrown unchanged so cancellation
 * never reads as an outage; the deadline is reported separately from an unreachable endpoint.
 * @param error - the rejection fetch produced.
 * @param signal - the caller's cancellation signal.
 * @param timedOut - whether this provider's own deadline fired.
 * @returns never; always throws.
 */
function throwTransportFailure(error: unknown, signal: AbortSignal, timedOut: boolean): never {
  if (signal.aborted) throw signal.reason
  if (timedOut) throw new TranscriptionError('provider-timeout', 'transcription endpoint did not answer in time', { cause: error })
  throw new TranscriptionError('provider-unavailable', 'transcription endpoint is unreachable', { cause: error })
}

/** Transcription over an OpenAI-compatible HTTP endpoint. */
export class OpenAiCompatibleTranscription extends TranscriptionEngine {
  static inject = ['credentials']

  /** Loader validation for the endpoint, model, credential reference, and deadline. */
  static Config: z<Config> = z.object({
    baseUrl: z.string().required(),
    model: z.string().required(),
    apiKeyEnv: z.string(),
    timeoutMs: z.number().step(1).min(1).required(),
  })

  private readonly baseUrl: string
  private readonly model: string
  private readonly timeoutMs: number
  private readonly apiKeyRef?: CredentialRef

  /**
   * @param ctx - registrant context carrying the credential seam.
   * @param config - the validated endpoint, model, credential reference, and deadline.
   */
  constructor(ctx: Context, config: Config) {
    super(ctx)
    this.baseUrl = normalizeBaseUrl(config.baseUrl)
    this.model = config.model
    this.timeoutMs = config.timeoutMs
    if (config.apiKeyEnv !== undefined) this.apiKeyRef = credentialRef(config.apiKeyEnv)
  }

  /**
   * Report identity and whether the configured credential currently resolves.
   * @returns the provider's identity, model, accepted media types, and readiness.
   */
  async describe(): Promise<TranscriptionProviderInfo> {
    const base = {
      provider: PROVIDER_NAME,
      model: this.model,
      acceptedMediaTypes: ACCEPTED_MEDIA_TYPES,
    }
    if (this.apiKeyRef === undefined) return { ...base, ready: true }
    const info = await this.ctx.credentials.describe(this.apiKeyRef)
    return info.configured
      ? { ...base, ready: true }
      : { ...base, ready: false, detail: `no value for ${this.apiKeyRef}` }
  }

  /**
   * Post one clip as multipart form data and return the endpoint's text.
   * @param clip - the complete encoded recording.
   * @param signal - caller-owned cancellation.
   * @returns the recognized text and the language the endpoint reported.
   */
  async transcribe(clip: AudioClip, signal: AbortSignal): Promise<TranscriptResult> {
    if (clip.data.byteLength === 0) throw new TranscriptionError('empty-audio', 'clip carries no audio')
    const extension = mediaTypeExtension(clip.mimeType)
    const media = bareMediaType(clip.mimeType)

    const headers: Record<string, string> = {}
    if (this.apiKeyRef !== undefined) {
      const hit = await this.ctx.credentials.resolve(this.apiKeyRef)
      if (hit === undefined) {
        throw new TranscriptionError('not-configured', `no value for ${this.apiKeyRef}`)
      }
      headers['authorization'] = `Bearer ${hit.value}`
    }

    const form = new FormData()
    form.set('model', this.model)
    form.set('file', new Blob([clip.data], { type: media }), `clip.${extension}`)
    if (clip.language !== undefined) form.set('language', clip.language)

    const timeout = AbortSignal.timeout(this.timeoutMs)
    let response: Response
    try {
      response = await fetch(`${this.baseUrl}/audio/transcriptions`, {
        method: 'POST',
        headers,
        body: form,
        signal: AbortSignal.any([signal, timeout]),
      })
    } catch (error) {
      throwTransportFailure(error, signal, timeout.aborted)
    }

    if (!response.ok) throw classifyHttpFailure(response.status, await readErrorBody(response, signal, timeout))

    let payload: unknown
    try {
      payload = await response.json()
    } catch (error) {
      if (signal.aborted || timeout.aborted) throwTransportFailure(error, signal, timeout.aborted)
      throw new TranscriptionError('provider-rejected', 'endpoint returned a non-JSON body', { cause: error })
    }
    return parseTranscriptionBody(payload)
  }
}

export default OpenAiCompatibleTranscription
