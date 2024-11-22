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
    rev = "v0.11.0";
    hash = "sha256:1yf7q1v2zy77bxmx4q5b1pdma6rc3qx4fv5jx8bsl7ad4vkldn05";
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
