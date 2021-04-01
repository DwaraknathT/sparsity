.PHONY: deps_table_update style 
check_dirs := src

# Format source code automatically and check is there are any problems left that need manual fixing

style: deps_table_update
	black $(check_dirs)
	isort $(check_dirs)
