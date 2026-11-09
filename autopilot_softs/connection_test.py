import krpc

connection = krpc.connect(name="Luna-16 connection test")

status = connection.krpc.get_status()
vessel = connection.space_center.active_vessel

print(f"kRPC version: {status.version}")
print(f"Active vessel: {vessel.name}")

connection.close()