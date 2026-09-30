import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from jobs import run_owned, stream_owned
import torch
from runtime import RollingOverlap, SharedNeuTTS
from neutts import NeuTTS
from neutts.neutts import _linear_overlap_add


class OverlapTests(unittest.TestCase):
    def test_matches_full_history_and_bounds_memory(self):
        rng = np.random.default_rng(17)
        stride = 12000
        frames = [rng.normal(size=12960).astype(np.float32) for _ in range(30)]
        frames.append(rng.normal(size=3743).astype(np.float32))
        rolling = RollingOverlap(stride)
        produced = []
        for i, frame in enumerate(frames):
            final = i == len(frames) - 1
            produced.append(rolling.add(frame, final=final))
            if not final:
                self.assertLessEqual(rolling.out.shape[-1], 960)
        expected = _linear_overlap_add(frames, stride)
        np.testing.assert_array_equal(np.concatenate(produced), expected)

    def test_final_window_shorter_than_existing_overlap(self):
        frames = [np.ones(12960, np.float32), np.ones(300, np.float32) * 2]
        rolling = RollingOverlap(12000)
        actual = np.concatenate([rolling.add(frames[0]), rolling.add(frames[1], final=True)])
        np.testing.assert_array_equal(actual, _linear_overlap_add(frames, 12000))


class ReferenceCacheTests(unittest.TestCase):
    def test_cache_hits_skip_encoder_and_invalid_or_changed_audio_reencode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, semantic = root / 'codec.bin', root / 'semantic.bin'
            checkpoint.write_bytes(b'codec-weights')
            semantic.write_bytes(b'encoder-weights')
            a, b = root / 'a.wav', root / 'b.wav'
            a.write_bytes(b'voice-a')
            b.write_bytes(b'voice-b')
            tts = SharedNeuTTS.__new__(SharedNeuTTS)
            decoder = SimpleNamespace(device=torch.device('cpu'))
            tts.codec = decoder
            tts.codec_repo = 'neuphonic/neucodec'
            tts.codec_checkpoint = str(checkpoint)
            tts._reference_tokens = {}
            with patch('runtime.cached_file', return_value=str(semantic)), \
                 patch('runtime.importlib.metadata.version', return_value='test'), \
                 patch('runtime.NeuCodec.from_pretrained') as encoder, \
                 patch.object(NeuTTS, 'encode_reference', return_value=torch.tensor([2, 4, 6])) as encode:
                references = {'a': str(a), 'b': str(b)}
                self.assertEqual(tts.references(references, root / 'cache'), {'a': [2, 4, 6], 'b': [2, 4, 6]})
                self.assertEqual(encode.call_count, 2)
                self.assertIs(tts.codec, decoder)
                encoder.reset_mock()
                encode.reset_mock()
                tts.references(references, root / 'cache')
                encoder.assert_not_called()
                encode.assert_not_called()
                # A corrupt cache re-encodes only its affected reference.
                next((root / 'cache').glob('*.json')).write_text('[true]')
                tts.references(references, root / 'cache')
                self.assertEqual(encode.call_count, 1)
                encode.reset_mock()
                a.write_bytes(b'updated-voice-a')
                tts.references(references, root / 'cache')
                self.assertEqual(encode.call_count, 1)
                self.assertIs(tts.codec, decoder)


class OwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.lock = asyncio.Lock()
        # Multiple threads intentionally expose defects masked by production's one-thread pool.
        self.executor = ThreadPoolExecutor(max_workers=2)

    async def asyncTearDown(self):
        self.executor.shutdown(wait=True)

    async def test_two_streams_never_overlap(self):
        running = 0
        peak = 0
        guard = threading.Lock()

        def produce(cancelled, emit):
            nonlocal running, peak
            with guard:
                running += 1
                peak = max(peak, running)
            try:
                for _ in range(8):
                    if not emit(b'a'):
                        return
                    time.sleep(0.005)
            finally:
                with guard:
                    running -= 1

        async def consume():
            return [item async for item in stream_owned(self.lock, self.executor, produce)]
        a, b = await asyncio.gather(consume(), consume())
        self.assertEqual((len(a), len(b), peak), (8, 8, 1))

    async def test_disconnect_releases_worker_before_ownership(self):
        stopped = threading.Event()
        emitted = []

        def produce(cancelled, emit):
            try:
                while emit(b'x'):
                    emitted.append(1)
            finally:
                # Ensure cancellation cleanup is actually waited for.
                time.sleep(0.02)
                stopped.set()

        stream = stream_owned(self.lock, self.executor, produce)
        self.assertEqual(await anext(stream), b'x')
        await asyncio.sleep(0.05)
        self.assertLessEqual(len(emitted), 5)
        await stream.aclose()
        self.assertTrue(stopped.is_set())
        self.assertFalse(self.lock.locked())

    async def test_task_cancellation_waits_for_stream_worker(self):
        stopped = threading.Event()

        def produce(cancelled, emit):
            try:
                emit(b'a')
                cancelled.wait(2)
            finally:
                stopped.set()

        async def consume():
            async for _ in stream_owned(self.lock, self.executor, produce):
                pass
        task = asyncio.create_task(consume())
        await asyncio.sleep(0.03)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(stopped.is_set())
        self.assertFalse(self.lock.locked())

    async def test_nonstream_cancellation_keeps_ownership(self):
        stopped = threading.Event()

        def produce(cancelled):
            cancelled.wait(2)
            time.sleep(0.02)
            stopped.set()
            return b'a'
        task = asyncio.create_task(run_owned(self.lock, self.executor, produce))
        await asyncio.sleep(0.03)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(stopped.is_set())
        self.assertFalse(self.lock.locked())

    async def test_worker_failure_is_delivered_and_lock_released(self):
        def produce(cancelled, emit):
            raise ValueError('decoder failed')
        with self.assertRaisesRegex(ValueError, 'decoder failed'):
            async for _ in stream_owned(self.lock, self.executor, produce):
                pass
        self.assertFalse(self.lock.locked())


if __name__ == '__main__':
    unittest.main()
