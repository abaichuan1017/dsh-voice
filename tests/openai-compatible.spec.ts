import { createServer } from 'node:http'
import type { AddressInfo } from 'node:net'
import { describe, expect, it, vi } from 'vitest'
import { OpenAiCompatibleTranscription } from '../src/providers/openai-compatible.ts'

const clip = { data: new Uint8Array([1, 2]), mimeType: 'audio/wav' }

/** Exercise real fetch body reads against a loopback endpoint, without credentials or audio services. */
async function endpoint(
  status: number,
  body: string | undefined,
  run: (baseUrl: string) => Promise<void>,
): Promise<void> {
  const server = createServer((request, response) => {
    request.resume()
    response.writeHead(status, { 'content-type': 'application/json' })
    if (body === undefined) response.flushHeaders()
    else response.end(body)
  })
  await new Promise<void>((resolve) => { server.listen(0, '127.0.0.1', resolve) })
  const { port } = server.address() as AddressInfo
  try {
    await run(`http://127.0.0.1:${port}/v1`)
  } finally {
    server.closeAllConnections()
    await new Promise<void>((resolve, reject) => { server.close(error => error ? reject(error) : resolve()) })
  }
}

/** As in the whisper.cpp tests, bypass the constructor's Cordis registration. */
function provider(baseUrl: string, timeoutMs = 10_000): OpenAiCompatibleTranscription {
  return Object.assign(Object.create(OpenAiCompatibleTranscription.prototype) as object, {
    baseUrl, model: 'test', timeoutMs,
  }) as unknown as OpenAiCompatibleTranscription
}

describe('OpenAiCompatibleTranscription response bodies', () => {
  for (const status of [200, 503]) {
    it(`reports a deadline while reading a ${status} body as a timeout`, async () => {
      await endpoint(status, undefined, async (baseUrl) => {
        await expect(provider(baseUrl, 50).transcribe(clip, new AbortController().signal))
          .rejects.toMatchObject({ code: 'provider-timeout' })
      })
    })

    it(`preserves cancellation while reading a ${status} body`, async () => {
      const fetch = globalThis.fetch
      let started!: () => void
      const bodyStarted = new Promise<void>((resolve) => { started = resolve })
      const spy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (...args) => {
        const response = await fetch(...args)
        const method = status === 200 ? 'json' : 'text'
        const read = response[method].bind(response)
        vi.spyOn(response, method).mockImplementation(() => { started(); return read() })
        return response
      })
      try {
        await endpoint(status, undefined, async (baseUrl) => {
          const controller = new AbortController()
          const pending = provider(baseUrl).transcribe(clip, controller.signal)
          await bodyStarted
          const reason = new Error('preview no longer needed')
          controller.abort(reason)
          await expect(pending).rejects.toBe(reason)
        })
      } finally {
        spy.mockRestore()
      }
    })
  }

  it('still rejects malformed JSON as a provider response error', async () => {
    await endpoint(200, 'not JSON', async (baseUrl) => {
      await expect(provider(baseUrl).transcribe(clip, new AbortController().signal))
        .rejects.toMatchObject({ code: 'provider-rejected', message: 'endpoint returned a non-JSON body' })
    })
  })

  it('still returns a completed transcript', async () => {
    await endpoint(200, '{"text":" hello "}', async (baseUrl) => {
      await expect(provider(baseUrl).transcribe(clip, new AbortController().signal))
        .resolves.toEqual({ text: 'hello' })
    })
  })

  it('still keeps the diagnostic from a completed HTTP error response', async () => {
    await endpoint(503, 'upstream unavailable', async (baseUrl) => {
      await expect(provider(baseUrl).transcribe(clip, new AbortController().signal))
        .rejects.toMatchObject({ code: 'provider-rejected', message: 'endpoint returned 503: upstream unavailable' })
    })
  })
})
