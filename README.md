## Cricket Run Expectancy per over

Running the code
1. [Install Docker](https://docs.docker.com/engine/install/) on your system
2. Make sure `match_results.json` and `innings_results.json` files in the `cric-run-expectancy/data` folder (available on [cricsheet](https://cricsheet.org/format/), also added in under Github Release), needed to be used by the `data_gen.script`
3. Follow instructions given below,
```
git clone https://github.com/sidthakur08/cric-run-expectancy.git
cd cric-run-expectancy
docker build -t cric-run-expectancy .
docker run cric-run-expectancy # runs data_gen, model, pred and tests
```
