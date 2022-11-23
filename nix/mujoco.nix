{ autoPatchelfHook
, fetchurl
, lib
, libGL
, stdenv
, xorg
, zlib
, libXxf86vm
}:

stdenv.mkDerivation rec {
  pname = "mujoco";
  version = "2.1.0"; # mujoco-py only supports up to version 2.1.0 of mujoco

  src = fetchurl {
    url = "https://github.com/deepmind/mujoco/releases/download/${version}/mujoco210-linux-x86_64.tar.gz";
    sha256 = "pDbKL0FEw4uDcgVjW71g/+EWLVtEyH3yIjJ5WXjX0BI=";
  };

  nativeBuildInputs = [
    autoPatchelfHook
  ];

  buildInputs = [
    zlib
    libGL
    xorg.libX11
    xorg.libXcursor
    xorg.libXext
    xorg.libXi
    xorg.libXinerama
    xorg.libXrandr
    stdenv.cc.cc.lib
    libXxf86vm
  ];

  dontBuild = true;

  installPhase = ''
    mkdir $out
    cp -r bin include model sample $out
    addAutoPatchelfSearchPath $out/include $out/sample
  '';

  meta = with lib; {
    description = "Multi-Joint dynamics with Contact. A general purpose physics simulator.";
    homepage = "https://mujoco.org/";
    license = licenses.unfree;
    architectures = "x86_64";
  };
}
