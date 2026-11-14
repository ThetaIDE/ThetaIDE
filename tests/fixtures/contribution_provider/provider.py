"""Independent extension imports no private host implementation."""
def activate(context):
    context.append('activated')
    return lambda: context.append('closed')
