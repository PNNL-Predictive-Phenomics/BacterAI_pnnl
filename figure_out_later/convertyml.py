import yaml

def convert_env_to_requirements(env_file="bacterai_env.yml", req_file="requirements.txt"):
    with open(env_file, 'r') as f:
        env_data = yaml.safe_load(f)

    pip_packages = []

    pip_deps = env_data.get('dependencies', [])
    for dep in pip_deps:
        # Handle pip-specific dict dependencies
        if isinstance(dep, dict) and 'pip' in dep:
            pip_packages.extend(dep['pip'])
        # Handle standalone dependencies (strings)
        elif isinstance(dep, str):
            pip_packages.append(dep)

    with open(req_file, 'w') as f:
        for package in pip_packages:
            f.write(f"{package}\n")


if __name__ == "__main__":
    convert_env_to_requirements()
