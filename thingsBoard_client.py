import time
import paho.mqtt.client as mqtt
import json

THINGSBOARD_HOST = '83.212.126.9'
ACCESS_TOKEN = 'YRAD7MrmP0JjuFC8VqGl'

# Data capture and upload interval in seconds. Less interval will eventually hang the DHT22.
INTERVAL=2

sensor_data = {'temperature': 0, 'humidity': 0}
sensor_dataGPS = {'latitude': 0, 'longitude': 0}

client = mqtt.Client()

# Set access token
client.username_pw_set(ACCESS_TOKEN)

# Connect to ThingsBoard using default MQTT port and 60 seconds keepalive interval
client.connect(THINGSBOARD_HOST, 1883, 60)

client.loop_start()

try:
    humidity, temperature = 20, 20
    latitude, longitude, speed = 46.1866, 14.5971, 60
    while True:

        humidity = round(humidity, 2)
        temperature = round(temperature, 2)
        print(u"Temperature: {:g}\u00b0C, Humidity: {:g}%".format(temperature, humidity))
        sensor_data['temperature'] = temperature
        sensor_data['humidity'] = humidity
        sensor_dataGPS['latitude'] = latitude
        sensor_dataGPS['longitude'] = longitude
        sensor_dataGPS['Speed'] = speed


        # Sending humidity and temperature data to ThingsBoard
        print(json.dumps(sensor_dataGPS))
        client.publish('v1/devices/me/telemetry', json.dumps(sensor_dataGPS), 1)
        client.publish('v1/devices/me/telemetry', json.dumps(sensor_data), 1)

        latitude = latitude + 0.0001
        longitude = longitude + 0.0001
        speed = speed + 5
        time.sleep(5)
except KeyboardInterrupt:
    pass

client.loop_stop()
client.disconnect()