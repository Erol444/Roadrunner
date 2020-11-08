import requests
ip = 'http://localhost:8080'
#ip = 'http://83.212.126.9:8080'

def getEvents():
    r = requests.get(ip+'/getevents')
    print(r.text)

def sendScore(lat, long, score):
    r = requests.get(ip+'/score?lat='+str(lat)+'&long='+str(long)+'&score='+str(score))
    print(r.text)

def getScore():
    r = requests.get(ip+'/getscore')
    print(r.text)

# send event/image
def sendImg(lat, long, img):
    r = requests.post(ip+'/postimage',
        files={'image': ('image.png', img, 'application/' + img.name.split('.')[1], {'Expires': '0'})},
        data={'lat': str(lat), 'long': str(long)}
    )
    print(r.text)

# sendScore(45.13, 14.67, 33)
# getScore()

# Change this to RGB image!
with open('testImg.png', 'rb') as img:
    sendImg(45.13, 14.67, img)
