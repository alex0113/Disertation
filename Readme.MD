# Install neo4j
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/neo4j.gpg

echo 'deb [signed-by=/etc/apt/keyrings/neo4j.gpg] https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list

sudo apt update
sudo apt install neo4j

In /etc/neo4j/neo4j.conf:
server.browser.enabled=true
server.http.listen_address=0.0.0.0:7474

cypher-shell -u neo4j -p neo4j "ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO 'hug0nam1'"

# Run neo4j
systemctl enable neo4j.service
 systemctl start neo4j.service

neo4j status

 Login:
 http://localhost:7474/
 First credentials:
 Username: neo4j
 Password: neo4j