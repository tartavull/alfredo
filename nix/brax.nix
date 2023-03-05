{ buildPythonPackage
, pythonRelaxDepsHook
, fetchFromGitHub
, absl-py
, dm_env
, etils
, flask
, flask-cors
, flax
, grpcio
, gym
, jax
, jaxlib
, jaxopt
, jinja2
  #, mujoco
, numpy
, optax
, pillow
, pytinyrenderer
, scipy
, tensorboardx
, trimesh
, typing-extensions
}:


buildPythonPackage rec {
  name = "brax";
  src = fetchFromGitHub {
    owner = "google";
    repo = "brax";
    rev = "v0.1.1";
    hash = "sha256-HtdFkVK+QODA32s/I4CJB06i2g3Rr/IZyf7L45lc4Bo=";
  };

  nativeBuildInputs = [
    pythonRelaxDepsHook
  ];

  prePatch = ''
    substituteInPlace setup.py \
        --replace 'mujoco' ' ' \
        --replace 'dataclasses' ' '
  '';

  pythonRemoveDeps = [ "mujoco" ];

  doCheck = false;

  propagatedBuildInputs = [
    absl-py
    # dataclasses
    dm_env
    etils
    flask
    flask-cors
    flax
    grpcio
    gym
    jax
    jaxlib
    jaxopt
    jinja2
    numpy
    optax
    pillow
    pytinyrenderer
    scipy
    tensorboardx
    trimesh
    typing-extensions
    #mujoco
  ];
}
