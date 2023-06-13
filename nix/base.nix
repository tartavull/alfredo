{ config, pkgs, lib, ... }:

{
  imports = [
    "${builtins.fetchTarball {
      url = "https://github.com/nix-community/home-manager/archive/release-22.11.tar.gz";
      sha256 = "sha256:1cp2rpprcfl4mjsrsrpfg6278nf05a0mpl3m0snksvdalfmc5si5";
    }}/nixos"
  ];

  # Generic settings
  system.stateVersion = "22.11";

  # NVIDIA configuration
  nixpkgs.config.allowUnfree = true;
  services.xserver.videoDrivers = [ "nvidia" ];
  hardware.opengl.enable = true;

  nix.settings = {
    experimental-features = [ "nix-command" "flakes" ];
    trusted-users = [ "root" "dev" ];
    substituters = [
      "https://cuda-maintainers.cachix.org"
      "https://tartavull.cachix.org"
    ];
    trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
      "tartavull.cachix.org-1:xxmUheA3nzwan59bFhfKEShnPeDXMeii+sWHnbq8PsQ="
    ];
  };


  # Users
  users.users = {
    dev = {
      isNormalUser = true;
      extraGroups = [ "wheel" ];
    };
    root = {
      hashedPassword = "!";
    };
  };

  # Packages
  environment.systemPackages = with pkgs; [
    google-cloud-sdk
    git
    cachix
    vim
    htop
    screen
    home-manager
    nodejs
  ];

  home-manager.users.dev = {
    home.stateVersion = "22.11"; # add this line

    programs.neovim = {
      enable = true;
      vimAlias = true;
      extraConfig = ''
        set clipboard+=unnamedplus
        colorscheme gruvbox
        set tabstop=4 shiftwidth=4 expandtab'';
      # list all available plugins
      # nix-env -f '<nixpkgs>' -qaP -A vimPlugins
      plugins = with pkgs.vimPlugins; [
        vim-nix
        gruvbox
        colorizer
        coc-nvim
        coc-python
        coc-json
        coc-spell-checker
      ];
    };

    home.sessionVariables = {
      EDITOR = "vim";
    };

    home.file.".screenrc".text = ''
      altscreen on
      defscrollback 5000
    '';

    programs.bash = {
      enable = true;
      initExtra = ''
        if [[ -n $SSH_CONNECTION ]]; then
          if [ ! -d "$HOME/alfredo" ]; then
            git clone https://github.com/tartavull/alfredo.git "$HOME/alfredo"
          fi
          if [[ -z $STY ]]; then
            # Get the id of the first detached screen session
            first_detached=$(screen -ls | grep "(Detached)" | head -n1 | awk '{print $1}')
            # If a detached session is found, reattach it
            if [[ -n $first_detached ]]; then
              exec screen -r $first_detached
            # If no detached session is found, start a new one
            else
              exec screen -R
            fi
          fi
        fi
      '';
    };
  };
}
