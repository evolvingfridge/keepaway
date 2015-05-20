mkdir logs

# modify agent.env file (insert env variables to agent)

# to run docker
docker run -d --env-file=agent.env -v $(pwd)/logs:/home/soccer/logs --name=keepaway_soccer mkurek/keepaway:latest

# to stop calculation
docker stop keepaway_soccer
