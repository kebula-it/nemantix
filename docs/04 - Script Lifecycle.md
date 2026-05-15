# Script Lifecycle

Lifecycle of a Nemantix script:
<p align="center">
  <img src="images/flow.png" alt="Nemantix workflow" height="250"/>
</p>

* **NXS**: This is the initial **development phase** where the users write `.nxs` files.
An NXS script can have partially specified actions and deliberates. If this is the case,
it cannot be executed as it is, since it needs to be "completed" first. Otherwise,
execution can occur.
* **NXC**: NXS scripts that need completion (e.g., there are `undefined`, `drafted`, or `editable` actions or plans)
are coded and then executed: this is what we call **coding**. During execution, coding can happen if the
previous criteria are met. Once the NXC is defined, i.e., a stable executable behavior is accomplished, the
**testing phase** is concluded.
* **NXV**: To ensure security and reliability the final NXC script is signed, resulting in a unified `.nxv` format.
Signing produces a signature that is checked for authenticity before execution. In this way, we prevent users
(even malicious ones) to edit the source files with unapproved modifications.

### Sign a script
The signing of a Nemantix script occurs as follows:
1. As first step, we need to generate a private and public key pair.
```python
# key-pair generation
from nemantix.security.ecdsa import generate_keys

generate_keys(base_path='my_folder')

# it generates:
# - nmx_ecdsa_private.pem and nmx_ecdsa_private.pem files
```
2. We use the private key to sign a script:
```python
from nemantix.security import Signer
from nemantix.core.script import Script
from nemantix.core.source_manager import LocalSourceManager

script = Script('path-to-my/script.nxc', source_manager=LocalSourceManager())

signer = Signer(private_key_path='my_folder/nmx_ecdsa_private.pem')
signed_script = signer.sign(script)
```

The source code of the `signed_script` (which is written at the same path of `script` but with `.nxv` extension)
includes a special comment containing the signature:
```bash
# NXV-SIGN:  304502200eceb6f27a680b20f4c5c132c81939...
deliberate MyDeliberate when >> ... <<:
    plan:
        ...
    __plan
__deliberate
```
The signature is then extracted during verification and matched against the source code to verify its authenticity.

### Verification
Verification uses the public key (e.g., `nmx_ecdsa_public.pem`) to verify a given script:
```python
from nemantix.security import Verifier

verifier = Verifier(public_key_path='my_folder/nmx_ecdsa_public.pem')

if verifier.verify(signed_script):
    print('Verified.')
else:
    print('Invalid signature!')
```

Next: [Toolsets](./05%20-%20Toolsets.md)
