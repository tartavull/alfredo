{ buildPythonPackage
, fetchFromGitHub
, jax
, jaxlib
, orbax
, rich
, optax
, matplotlib
}:

buildPythonPackage rec {
  name = "flax";
  format = "pyproject";
  src = fetchFromGitHub {
    owner = "google";
    repo = "flax";
    rev = "v0.7.2";
    hash = "sha256-Zj2xwtUBYrr0lwSjKn8bLHiBtKB0ZUFif7byHoGSZvg=";
  };
  propagatedBuildInputs = [
    jax
    jaxlib
    orbax
    rich
    optax
    matplotlib
  ];
  postPatch = ''
    sed -i '/tensorstore/d' pyproject.toml
    sed -i '/orbax/d' pyproject.toml
  '';
  doCheck = false;
}
