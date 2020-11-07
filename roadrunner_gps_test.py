import http.server
import socketserver
import threading
import time
from urllib.parse import urlparse, parse_qs

gps_lat = 0
gps_long = 0

class ServerHandler(http.server.BaseHTTPRequestHandler):

    def do_GET(self):
        print('GET REQUETS received')
        global gps_lat
        global gps_long
        
        url = urlparse(self.path)
        if url.path == "/gps" :
            url.query["gps"]


        #return http.server.BaseHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        content_len = int(self.headers['Content-Length'])
        post_body = self.rfile.read(content_len)
        print('post_body: ', post_body)
        #return http.server.BaseHTTPRequestHandler.do_POST(self)

        self.send_response(200)
        self.send_header("Set-Cookie", "foo=bar")
        self.end_headers()
        #self.wfile.write('')
        
def main():
    port = 5000
    with socketserver.TCPServer(("", port), ServerHandler) as httpd:
        print("serving at port", port)
        httpd.serve_forever()

th = threading.Thread(target=main)
th.daemon = True
th.start()

while True:
    print('hai', test)
    time.sleep(1)
