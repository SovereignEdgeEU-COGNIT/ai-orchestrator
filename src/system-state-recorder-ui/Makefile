all: container 
.PHONY: all container 

BUILD_IMAGE ?= johan/congitssr
PUSH_IMAGE ?= johan/cognitssr

container:
	docker build -t $(BUILD_IMAGE) .

push:
	docker tag $(BUILD_IMAGE) $(PUSH_IMAGE) 
	docker push $(PUSH_IMAGE)
