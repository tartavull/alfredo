{ buildPythonPackage
, fetchFromGitHub
, gym
, wandb
, stable-baselines
, core-go
}:

buildPythonPackage rec {
  name = "genetic-intelligence";
  src = ../src/python/.;
  propagatedBuildInputs = [ gym wandb stable-baselines core-go];
  GICORE="${core-go}/core.so";
}
