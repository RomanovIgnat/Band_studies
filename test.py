from pathlib import Path

from pymatgen.io.vasp.outputs import Vasprun, Outcar

vasprun_directory = Path("VASP/hBN_spin_500/poscar_BN_B56C4N60_4dfae1f9-c2bb-4789-8b21-baabcad7ff21")
vasp_folder = vasprun_directory / '01_relax'
vasprun_output = vasp_folder / 'vasprun.xml'

vasprun_file = Vasprun(
            str(vasprun_output),
            parse_potcar_file=False,
            separate_spins=True,
            parse_dos=True
        )
outcar = Outcar(str(vasp_folder / "OUTCAR"))

print(vasprun_file.efermi, outcar.efermi)
