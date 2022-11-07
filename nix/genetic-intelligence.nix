{ buildPythonPackage
, fetchFromGitHub
, gym
, wandb
, stable-baselines
}:

buildPythonPackage rec {
  name = "genetic-intelligence";
  src = ../.;
  propagatedBuildInputs = [ gym wandb stable-baselines];
}
