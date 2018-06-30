# English to Cypher
A model to transform english into Cypher queries, based off the [CLEVR graph dataset](https://github.com/Octavian-ai/clevr-graph).

This model seeks to transform sentences like:

> How many stations are between Koof Lane and Gag Street?

into equivalent Cypher statements:

> MATCH ( var1 ) MATCH ( var2 ) MATCH tmp1 = shortestPath ( ( var1 ) - [ * ] - ( var2 ) ) WHERE var1.name = " Koof Lane " AND var2.name = " Gag Street " WITH nodes ( tmp1 ) AS var3 RETURN length ( var3 ) - 2


## How to run this

From the root directory, first install the pre-requisites:
```shell
pipenv install
pipenv shell
```

All the command line python invocations assume `pipenv shell`

### Train

Then from the virtual-environment pipenv has provided, run the training:
```shell
python -m e2c.train
```

Training is quite slow without a GPU. If you don't happen to have a NVIDIA Titan under your desk, we've formatted this project to easily fun on Floyd Hub:

```shell
sudo pip install -U floyd-cli
floyd login
./floyd-train.sh
```

### Predict

Once you have your shiney trained model (or, just use ours!) you can ask questions and get answers from a real-deal Neo4j graph:

First, start a neo4j database locally (you need Docker for this script, or you can give alternative arguments to another neo4j instance): `./start-neo4j-database.sh`

Then run the prediction programme:

```shell
$ python -m e2c.predict 

Example questions:
> Which lines is Proy Palace on?
> How many lines is Snosk Boulevard on?
> How clean is Grir Court?
> Are Plir International and Swongton on the same line?
> Which stations does Cyan Soost pass through?

Example stations from graph:
> Draz Boulevard, Strov Boulevard, Swuct Hospital, Fak Boulevard, Frook Lane

Example lines from graph:
> Green Soosh, Green Fliv, Olive Huw, Purple Sweb, Blue Prooy

Ask a question:
```

## Dataset generation

This project takes the data from CLEVR graph and extracts the Question-Cypher pairs into simple text files. The `src.txt` and `tgt.txt` files have one line per translation pair (with a test and training set), and `vocab.txt` contains every word used in all of the language files.

You can easily re-generate the language files (e.g. from your own GQA YAML):
```shell
python -m e2c.build_data
```



## Acknowledgements

Big shout out to [the TensorFlow NMT tutorial](https://github.com/tensorflow/nmt) which I've heavily based this on