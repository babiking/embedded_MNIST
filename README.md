# embedded_MNIST
embed MNIST classification algorithm into an IoT device

# Install NVIDIA Graphic Driver
	1. Blacklist nouveau driver
		sudo vim /etc/modprobe.d/blacklist-nouveau.conf
		then,
			"blacklist nouveau
			options modset=0"
	2. Run the command "sudo update-initramfs -u"
	3. Check if the nouveau driver is still working
		lspci | grep nouveau
	4. Stop X-server by running the command "sudo service lightdm stop"
	5. Install the NVIDIA Graphic Driver
		./NVIDIA.run -no-x-check -no-nouveau-check -no-opengl-check
	6. Turn off the UEFI Boot Mode and corresponding Security Boot in BIOS
