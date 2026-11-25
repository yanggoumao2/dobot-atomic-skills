"""List USB identities without opening devices or modifying configuration."""
from serial.tools import list_ports
if __name__ == "__main__":
    for p in list_ports.comports():
        print(f"{p.device} | {p.description} | serial={p.serial_number} | {p.hwid}")
