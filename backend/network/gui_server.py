import sys
import threading
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, HTTPServer
import json

from PySide6.QtCore import QObject
from PySide6.QtWebSockets import QWebSocketServer
from PySide6.QtWebChannel import QWebChannel, QWebChannelAbstractTransport

class WebSocketTransport(QWebChannelAbstractTransport):
    def __init__(self, socket, parent=None):
        super().__init__(parent)
        self.socket = socket
        self.socket.textMessageReceived.connect(self.textMessageReceived)
        self.socket.disconnected.connect(self.deleteLater)

    def sendMessage(self, message):
        self.socket.sendTextMessage(json.dumps(message))

    def textMessageReceived(self, message):
        self.messageReceived.emit(json.loads(message), self)

class GUIServer(QObject):
    def __init__(self, bridge_obj, frontend_dir, http_port=8000, ws_port=8001, parent=None):
        super().__init__(parent)
        self.frontend_dir = frontend_dir
        self.http_port = http_port
        self.ws_port = ws_port
        
        # Setup WebSocket Server for QWebChannel
        self.ws_server = QWebSocketServer("QWebChannel Server", QWebSocketServer.NonSecureMode, self)
        if not self.ws_server.listen(port=self.ws_port):
            print(f"Failed to start WebSocket server on port {self.ws_port}")
            
        self.channel = QWebChannel(self)
        self.channel.registerObject("backend", bridge_obj)
        
        self.ws_server.newConnection.connect(self.on_new_connection)
        
        # Setup HTTP Server
        self.http_server = None
        self.http_thread = None
        self.start_http_server()

    def on_new_connection(self):
        socket = self.ws_server.nextPendingConnection()
        transport = WebSocketTransport(socket, socket)
        self.channel.connectTo(transport)

    def start_http_server(self):
        # Local variables to capture for the closure
        frontend_dir = self.frontend_dir
        
        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(frontend_dir), **kwargs)
                
            def log_message(self, format, *args):
                pass
                
        self.http_server = HTTPServer(('0.0.0.0', self.http_port), Handler)
        self.http_thread = threading.Thread(target=self.http_server.serve_forever, daemon=True)
        self.http_thread.start()
        print(f"LAN GUI available at http://0.0.0.0:{self.http_port}")

    def shutdown(self):
        if self.http_server:
            self.http_server.shutdown()
            self.http_server.server_close()
        if self.ws_server:
            self.ws_server.close()
