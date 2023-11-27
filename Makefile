.PHONY: up down clean

clean: down
	rm -f logpath/*

down: 
	docker compose down --volumes

up:
	docker compose up --build

