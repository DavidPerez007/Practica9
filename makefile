
IMAGE_NAME=adaboost-api
CONTAINER_NAME=adaboost-container
PORT=8000
TEAM_NAME=equipo_boosters


build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -d --name $(CONTAINER_NAME) -p $(PORT):8000 $(IMAGE_NAME)

status:
	docker ps -a | grep $(CONTAINER_NAME) || echo "El contenedor no existe."

stop:
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)


clean:
	docker system prune -f


package:
	tar --exclude='*.pyc' --exclude='__pycache__' --exclude='.venv' --exclude='.git' -czf $(TEAM_NAME).tar.gz .
