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
, orbax-checkpoint
}:


buildPythonPackage rec {
  name = "brax";
  src = fetchFromGitHub {
    owner = "google";
    repo = "brax";
    rev = "v0.10.5";
    hash = "sha256-Ek1j/tghkNOny6uPWM+WHlTB3eZI5yl3oXq4DdIEJv4=";
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
    orbax-checkpoint
  ];
}
