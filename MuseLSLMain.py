from muselsl import stream, list_muses

if __name__ == "__main__":

    muses = list_muses()
    if muses is None: muses = input("enter the muse MAC address from BlueMuse")
    address = "00:55:da:b7:44:b0"
    if not muses:
        print('No Muses found')
    else:
        if isinstance(muses, str):

            stream(address)
        else:
            stream(muses[0]['address'])

            # Note: Streaming is synchronous, so code here will not execute until the stream has been closed
        print('Stream has ended')
