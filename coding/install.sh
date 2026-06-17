if [ -d "../DOTA_devkit" ]; then
    cp -r ../DOTA_devkit ./
elif [ -d "../../DOTA_devkit" ]; then
    cp -r ../../DOTA_devkit ./
else
    echo "Error: DOTA_devkit not found in expected locations." >&2
fi
# cp ../Arial.ttf ./
chmod +x ./DOTA_devkit/polyiou_cpu/build.sh
# ./DOTA_devkit/polyiou_cpu/build.sh

