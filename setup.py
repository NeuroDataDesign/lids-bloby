import sys
import os
import subprocess

def _install_requirements():
	print('Installing Bloby requirements...')
	subprocess.run(['pip3', 'install', '-r', 'requirements.txt'])

def install():
	_install_requirements()

	from colorama import init as color_init
	from colorama import Fore
	from pyfiglet import Figlet

	color_init()

	print(Fore.GREEN + 'Bloby setup successful!')
	f = Figlet(font='slant')
	print(f.renderText('Bloby'))

if __name__ == '__main__':
	task = sys.argv[1]

	if task == 'install':
		install()
	elif task == 'clean':
		clean()
	else:
		print(Fore.RED + '`{}` not supported'.format(task))
