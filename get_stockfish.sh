wget https://stockfishchess.org/files/stockfish_15.1_linux_x64.zip
unzip stockfish_15.1_linux_x64
mv ./stockfish_15.1_linux_x64/stockfish-ubuntu-20.04-x86-64 stockfish
rm -r ./stockfish_15.1_linux_x64
rm stockfish_15.1_linux_x64.zip
chmod +x stockfish
