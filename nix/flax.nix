{ buildPythonPackage
, fetchFromGitHub
, jax
, jaxlib
, orbax
}:

buildPythonPackage rec {
  name = "flax";
  src = fetchFromGitHub {
    owner = "google";
    repo = "flax";
    rev = "v0.6.8";
    hash = "sha256-i+K1PUzsy4Xu7YzdZpMHwQ66KIOompp1X48bwOmKp14=";
  };
  propagatedBuildInputs = [
    jax
    jaxlib
    orbax
  ];
  pythonRemoveDeps = [ "orbax" ];
}
