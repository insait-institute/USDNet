import yaml

with open("environment.yml", "r") as f:
    data = yaml.safe_load(f)

pip_packages = []

for dep in data.get("dependencies", []):
    if isinstance(dep, dict) and "pip" in dep:
        for pkg in dep["pip"]:
            if not str(pkg).strip().startswith("#"):
                pip_packages.append(pkg)

with open("pip_requirements.txt", "w") as f:
    for pkg in pip_packages:
        f.write(f"{pkg}\n")

print(f"Extracted {len(pip_packages)} pip packages.")
