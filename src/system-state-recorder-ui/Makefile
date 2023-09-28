all: container 
.PHONY: all container 

BUILD_IMAGE ?= cognit/demo
PUSH_IMAGE ?= cognit/demo

container:
	docker build -t $(BUILD_IMAGE) .

push:
	docker tag $(BUILD_IMAGE) $(PUSH_IMAGE) 
	docker push $(PUSH_IMAGE)
