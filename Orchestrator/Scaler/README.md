This is the code of the scaler within the Frontend VM of the ServerlessRuntime oneflow service developed for the COGNIT project




# Run the producer

0. Create virtual env (i.e: using UV)

1. SSH into the Frontend VM of the ServerlessRuntime oneflow-service running the ON Frontend VM.

```
> source .venv/bin/active

> uvicorn producer_app:app --host 0.0.0.0 --port 8000
```

Test from your local browser:
`https://<NGINX_PROXY_PUBLIC_IP>/serverless/<QUEUE_NAME>`

- The queue name must exists otherwise the API will return an error.

# Run the scaler RabbitMQ consumer

```
> source .venv/bin/active

> python main.py localhost rabbitadmin rabbitadmin
```

# RabbitMQ utilities
```
# List number of messages for each queue
rabbitmqadmin list queues

# Send a message to the "test_scaler" queue
rabbitmqadmin publish routing_key="test_scaler" payload="Hello from CLI!"

# Consume one message from the "test_scaler" queue
rabbitmqadmin get queue="test_scaler" count=1 ackmode=ack_requeue_false
```
