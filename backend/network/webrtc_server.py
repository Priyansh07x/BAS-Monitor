import asyncio
import json
import threading
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
import cv2

from backend.network.streamer import get_ip_streamer

class CameraVideoStreamTrack(VideoStreamTrack):
    """
    A video stream track that returns frames from the global IPStreamer.
    """
    def __init__(self):
        super().__init__()  # don't forget this!
        self.streamer = get_ip_streamer()

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        
        frame_cv = self.streamer.get_latest_frame()
        if frame_cv is None:
            # If no frame is available, yield a black frame or just wait
            await asyncio.sleep(0.05)
            # Create a 640x480 black frame
            import numpy as np
            frame_cv = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            frame_cv = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)

        frame = VideoFrame.from_ndarray(frame_cv, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base

        # Pace the stream properly according to the monotonic clock internally handled by aiortc
        return frame

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    # Add a video track
    pc.addTrack(CameraVideoStreamTrack())

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("WebRTC Connection state is", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )

def _run_webrtc(port=8555):
    app = web.Application()
    
    import aiohttp_cors
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })
    
    resource = cors.add(app.router.add_resource("/offer"))
    cors.add(resource.add_route("POST", offer))

    runner = web.AppRunner(app)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, '0.0.0.0', port)
    loop.run_until_complete(site.start())
    print(f"WebRTC Signaling server running on http://0.0.0.0:{port}")
    loop.run_forever()

class WebRTCServer:
    def __init__(self, port=8555):
        self.port = port
        self.thread = threading.Thread(target=_run_webrtc, args=(port,), daemon=True)
        
    def start(self):
        self.thread.start()
