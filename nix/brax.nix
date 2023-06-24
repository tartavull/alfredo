{ buildPythonPackage
, pythonRelaxDepsHook
, fetchFromGitHub
, absl-py
, dm_env
, etils
, flask
, flask-cors
, grpcio
, gym
, jax
, jaxlib
, jaxopt
, numpy
, optax
, pillow
, pytinyrenderer
, scipy
, trimesh
, tensorboardx
, typing-extensions
, flax
, mujoco
}:


buildPythonPackage rec {
  name = "brax";
  src = fetchFromGitHub {
    owner = "google";
    repo = "brax";
    rev = "v0.9.1";
    hash = "sha256-tFoTsz+EEd35nO39/owBBKbJG1LnAGUZBoOJkYVuwlI=";
  };

  nativeBuildInputs = [
    pythonRelaxDepsHook
  ];

  prePatch = ''
    substituteInPlace setup.py \
        --replace 'jax>=0.4.6' ' ' \
        --replace 'jaxlib>=0.4.6' ' '
  '';

  doCheck = false;

  propagatedBuildInputs = [
    absl-py
    # dataclasses
    dm_env
    etils
    flask
    flask-cors
    grpcio
    gym
    jax
    jaxlib
    jaxopt
    numpy
    pillow
    pytinyrenderer
    scipy
    tensorboardx
    trimesh
    typing-extensions
    optax
    mujoco
    flax
  ];
}
