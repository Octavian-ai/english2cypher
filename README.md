# English to Cypher

A model to transform english into Cypher queries, based off the [CLEVR graph dataset](https://github.com/Octavian-ai/clevr-graph).

This model seeks to transform sentences like:

> How many stations are between Koof Lane and Gag Street?

and turn them into answers such as 

> 3

by way of translating the English into equivalent Cypher statements:

> MATCH ( var1 ) MATCH ( var2 ) MATCH tmp1 = shortestPath ( ( var1 ) - [ * ] - ( var2 ) ) WHERE var1.name = " Koof Lane " AND var2.name = " Gag Street " WITH nodes ( tmp1 ) AS var3 RETURN length ( var3 ) - 2


## Running the code

### Prerequisites

From the root directory, first install the pre-requisites:
```shell
pipenv install
pipenv shell
```

All the command line python invocations assume `pipenv shell` has been previously invoked (E.g. that you are in the virtual environment with the required python modules)


### Predictions

The most fun way to see this code in action is to fire up predict mode. You can invoke it with `python -m e2c.predict`

You'll need to have a Neo4j database for the code to load then query (heads up: The code will delete everything in your provided database, then upload its own graph). The easiest way to do this is to run Docker, then use our script `./start-neo4j-database.sh` to create a database with the required extensions and authentication values that the code uses by default.

If you want to use a different database configuration, the arguments `--neo-user --neo-password --neo-url` will let you specify how to connect to it.

The predict script will automatically download a trained model and its vocab if you do not have that locally.

Here's the prediction program in action:

```shell
$ python -m e2c.predict
Example stations from graph:
> Draz Boulevard, Strov Boulevard, Swuct Hospital, Fak Boulevard, Frook Lane, Niwham, Dawbridge, Flip Bridge

Example lines from graph:
> Green Soosh, Green Fliv, Olive Huw, Purple Sweb, Blue Prooy, Blue Moss, Orange Hift, Pink Woog

Example questions:
> How clean is Fak Boulevard?
> How big is Dawbridge?
> What music plays at Swuct Hospital?
> What architectural style is Dawbridge?
> Does Flip Bridge have disabled access?
> Does Niwham have rail connections?
> How many architectural styles does Green Soosh pass through?
> How many music styles does Pink Woog pass through?
> How many sizes of station does Olive Huw pass through?
> How many stations playing classical does Blue Prooy pass through?
> How many clean stations does Green Fliv pass through?
> How many large stations does Olive Huw pass through?
> How many stations with disabled access does Green Soosh pass through?
> How many stations with rail connections does Blue Prooy pass through?
> Which lines is Niwham on?
> How many lines is Flip Bridge on?
> Are Dawbridge and Strov Boulevard on the same line?
> Which stations does Orange Hift pass through?

Ask a question: Does Niwham have rail connections?
Translation into cypher: 'MATCH (var1) WHERE var1.name="Niwham"  WITH 1 AS foo, var1.hasrail AS var2 RETURN var2'

Answer: None

Ask a question: Which stations does Orange Hift pass through?
Translation into cypher: 'MATCH ()-[var1]-()  MATCH (var2:LINE) WHERE var2.name="Orange Hift"  WITH 1 AS foo, var1, var2.id AS var3 WHERE var1.line_id = var3  MATCH (var4)-[var1]-() WITH 1 AS foo, var4 AS var5 WITH DISTINCT var5 as var6, 1 AS foo  WITH 1 AS foo, var6.name AS var7  RETURN var7'

Answer: Blel Boulevard, Dent Hospital, Chih Way, Smad Road, Gusk Square, Hum Upon Thames, Ploongwich, Guz Way, Chot Estate, Tump Hospital

Ask a question: What architectural style is Dawbridge?
Translation into cypher: 'MATCH (var1) WHERE var1.name="Dawbridge"  WITH 1 AS foo, var1.architecture AS var2 RETURN var2'

Answer: modernist
```



### Train

You can train the model yourself if you'd like. It takes about 1.5hrs running on a recent NVidia GPU.

First, build the text input data:
```shell
python -m e2c.build_data
```
Then run the training:
```shell
python -m e2c.train
```

Training is quite slow without a GPU. If you don't happen to have a NVIDIA Titan under your desk, we've formatted this project to easily run on Floyd Hub (and even give some nice stats):

```shell
sudo pip install -U floyd-cli
floyd login
./floyd-train.sh
```

During training you can see examples of its predictions in `output/`:

```yaml
beam:
- MATCH ()-[var1]-()  MATCH (var2:LINE) WHERE var2.name="Red Crend"  WITH 1 AS foo, var1, var2.id AS var3 WHERE var1.line_id = var3  MATCH (var4)-[var1]-() WHERE var4.architecture = "concrete"  WITH 1 AS foo, var4 AS var5 WITH DISTINCT var5 as var6, 1 AS foo  RETURN length(collect(var6))
- MATCH ()-[var1]-()  MATCH (var2:LINE) WHERE var2.name="Red Cred"  WITH 1 AS foo, var1, var2.id AS var3 WHERE var1.line_id = var3  MATCH (var4)-[var1]-() WHERE var4.architecture = "concrete"  WITH 1 AS foo, var4 AS var5 WITH DISTINCT var5 as var6, 1 AS foo  RETURN length(collect(var6))
- MATCH ()-[var1]-()  MATCH (var2:LINE) WHERE var2.name="Red Crind"  WITH 1 AS foo, var1, var2.id AS var3 WHERE var1.line_id = var3  MATCH (var4)-[var1]-() WHERE var4.architecture = "concrete"  WITH 1 AS foo, var4 AS var5 WITH DISTINCT var5 as var6, 1 AS foo  RETURN length(collect(var6))
- MATCH ()-[var1]-()  MATCH (var2:LINE) WHERE var2.name="Red Rrend"  WITH 1 AS foo, var1, var2.id AS var3 WHERE var1.line_id = var3  MATCH (var4)-[var1]-() WHERE var4.architecture = "concrete"  WITH 1 AS foo, var4 AS var5 WITH DISTINCT var5 as var6, 1 AS foo  RETURN length(collect(var6))
- MATCH ()-[var1]-()  MATCH (var2:LINE) WHERE var2.name="Red Crond"  WITH 1 AS foo, var1, var2.id AS var3 WHERE var1.line_id = var3  MATCH (var4)-[var1]-() WHERE var4.architecture = "concrete"  WITH 1 AS foo, var4 AS var5 WITH DISTINCT var5 as var6, 1 AS foo  RETURN length(collect(var6))
- MATCH ()-[var1]-()  MATCH (var2:LINE) WHERE var2.name="Red Crend"  WITH 1 AS foo, var1, var2.id AS var3 WHERE var1.line_id = var3  MATCH (var4)-[var1]-() WHERE var4.architecture = "conerete"  WITH 1 AS foo, var4 AS var5 WITH DISTINCT var5 as var6, 1 AS foo  RETURN length(collect(var6))
- MATCH ()-[var1]-()  MATCH (var2:LINE) WHERE var2.name="Red Crand"  WITH 1 AS foo, var1, var2.id AS var3 WHERE var1.line_id = var3  MATCH (var4)-[var1]-() WHERE var4.architecture = "concrete"  WITH 1 AS foo, var4 AS var5 WITH DISTINCT var5 as var6, 1 AS foo  RETURN length(collect(var6))
- MATCH ()-[var1]-()  MATCH (var2:LINE) WHERE var2.name="Red Crund"  WITH 1 AS foo, var1, var2.id AS var3 WHERE var1.line_id = var3  MATCH (var4)-[var1]-() WHERE var4.architecture = "concrete"  WITH 1 AS foo, var4 AS var5 WITH DISTINCT var5 as var6, 1 AS foo  RETURN length(collect(var6))
- MATCH ()-[var1]-()  MATCH (var2:LINE) WHERE var2.name="Red Crend"  WITH 1 AS foo, var1, var2.id AS var3 WHERE var1.line_id = var3  MATCH (var4)-[var1]-() WHERE var4.architecture = "crnece"  WITH 1 AS foo, var4 AS var5 WITH DISTINCT var5 as var6, 1 AS foo  RETURN length(collect(var6))
- MATCH ()-[var1]-()  MATCH (var2:LINE) WHERE var2.name="Red Crend"  WITH 1 AS foo, var1, var2.id AS var3 WHERE var1.line_id = var3  MATCH (var4)-[var1]-() WHERE var4.architecture = "concrete"  WITH 1 AS foo, var4 AS var5 WITH DISTINCT var5 as var6, 1 AS foo  RETURN length(collect(var6) WHERE var6ocollectc]e() WHERE var6.architecture = "concrete"  WITH 1
guided:
- MATCH ()-[var1]-()  MATCH (var2:LINE) WHERE var2.name="Red Crend"  WITH 1 AS foo, var1, var2.id AS var3 WHERE var1.line_id = var3  MATCH (var4)-[var1]-() WHERE var4.architecture = "concrete"  WITH 1 AS foo, var4 AS var5 WITH DISTINCT var5 as var6, 1 AS foo  RETURN length(collect(var6))
input: How many concrete stations are on the Red Crend line?
target:
- MATCH ()-[var1]-()  MATCH (var2:LINE) WHERE var2.name="Red Crend"  WITH 1 AS foo, var1, var2.id AS var3 WHERE var1.line_id = var3  MATCH (var4)-[var1]-() WHERE var4.architecture = "concrete"  WITH 1 AS foo, var4 AS var5 WITH DISTINCT var5 as var6, 1 AS foo  RETURN length(collect(var6))
```

In this structure, "target" is what the network should output (the ground truth), "beam" is an array of the networks predictions, "guided" is a semi-prediction mode where the network is given the target string and just asked to guess the next token.





## Acknowledgements

Thanks to Andrew Jefferson, Ashwath Salimath, Scott Dimond for their support, ideas and proof-reading.

Big shout out to [the TensorFlow NMT tutorial](https://github.com/tensorflow/nmt) which I've heavily based this code on, and to Google for sharing [their research](https://ai.google/research/pubs/pub45610).
