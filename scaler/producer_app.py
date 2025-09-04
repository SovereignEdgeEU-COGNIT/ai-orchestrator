# This script receive a queue name from NGINX Proxy and publish a message on RabbitMQ.
# If the queue does not exist, it will raise an error.


from fastapi import FastAPI, HTTPException
import pika
import json
from datetime import datetime

# --- CONFIG ---
RABBITMQ_HOST = 'localhost'

# --- APP ---
app = FastAPI()

@app.post("/process_queue/{queue_name}")
def create_message_in_queue(queue_name: str):
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
        channel = connection.channel()

        # KEY CHANGE: Check for queue existence without creating it.
        # passive=True will raise an exception if the queue does not exist.
        channel.queue_declare(queue=queue_name, passive=True)

        message_body = json.dumps({
            'source': 'cognit_client',
            'timestamp_utc': datetime.utcnow().isoformat()
        })

        channel.basic_publish(exchange='jobs_fanout', routing_key='', body=message_body)

        connection.close()
        print(f"Message published to queue '{queue_name}'")
        return {"status": "ok", "queue": queue_name}

    except pika.exceptions.ChannelClosedByBroker as e:
        raise HTTPException(status_code=404, detail=f"Queue '{queue_name}' not found.")
    except pika.exceptions.AMQPConnectionError:
        raise HTTPException(status_code=503, detail="Service Unavailable: Could not connect to RabbitMQ.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@app.get("/")
def read_root():
    return {"service": "RabbitMQ Producer API", "status": "active"}
