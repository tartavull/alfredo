{ lib
, python3Packages
, fetchFromGitHub
, pybind11
, eigen
, stdenv
}:

python3Packages.buildPythonPackage rec {
  pname = "ml_dtypes";
  version = "v0.1.0"; # specify the version of the package

  src = fetchFromGitHub {
    owner = "jax-ml";
    repo = "ml_dtypes";
    rev = version;
    hash = "sha256-cNUvpHDphd+7Zfok0AmdpIaXn2xiXl6jf7tJPD08Zq4=";
  };

  buildInputs = [ eigen ];
  nativeBuildInputs = [ pybind11 ];

  propagatedBuildInputs = with python3Packages; [
    numpy
  ];

  NIX_CFLAGS_COMPILE = [ "-std=c++17" "-DEIGEN_MPL2_ONLY" ];

  postUnpack = ''
    rm -r source/third_party/eigen
    ln -s ${eigen}/include/eigen3 source/third_party/eigen
  '';

  meta = with lib; {
    description = "ML dtypes is a Python library that provides data types used in Machine Learning.";
    homepage = "https://github.com/jax-ml/ml_dtypes";
    license = licenses.asl20; # specify the correct license
    maintainers = with maintainers; [ tartavull ];
  };
}
