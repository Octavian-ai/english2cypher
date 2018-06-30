#!/bin/bash

docker run -it -p 7474:7474 -p 7687:7687 --env NEO4J_ACCEPT_LICENSE_AGREEMENT=yes --env NEO4J_AUTH=neo4j/clegr-secrets andrewjefferson/myneo4j:3.4.1-enterprise-plus-apoc