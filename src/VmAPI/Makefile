BUILD_IMAGE ?= johan/ai_server
PUSH_IMAGE ?= johan/ai_server

build:
	docker build -t $(BUILD_IMAGE) .
push:	
	docker tag $(BUILD_IMAGE) $(PUSH_IMAGE)
	docker push $(PUSH_IMAGE)
run:
	docker run -it -p 4567:4567 $(PUSH_IMAGE) python3 /ai/main.py

shell:
	docker tag $(BUILD_IMAGE) $(PUSH_IMAGE)
	docker run -it -p 4567:4567 $(PUSH_IMAGE) bash

