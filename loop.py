import subprocess
i = 0
while True:
	i += 1
	print("\nLooping snake game: " + str(i))
	subprocess.call("python game.py", shell=True)