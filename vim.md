# VIM/NEOVIM

## colors in tmux
1. run `:checkhealth` and follow the steps
    1. *for tmux* paste the following in .tmux.config
        ```sh
        set -g default-terminal "tmux-256color"
        set-option -sa terminal-overrides ',xterm:RGB'
        ```

## number of occurences in a file
```vim
:%s/<pattern>//n
```
