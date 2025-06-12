# DERify

**DERify** is a Python-based toolkit for performing Distributed Energy Resource (DER) interconnection studies and impact analysis on electric distribution networks. Built on top of the [dss-python](https://github.com/dss-extensions/dss_python) extension, DERify leverages multiple OpenDSS engine instances for fast, scalable scenario evaluation—making it ideal for utilities, consultants, and researchers.

## Features

- **Automated Loss and Fault Studies**  
  Evaluate how DERs, such as battery storage and PV systems, impact network losses and fault currents.

- **Multi-Engine, Parallel Scenario Analysis**  
  Harness multiple OpenDSS engines in parallel to quickly evaluate large batches of scenarios or system configurations.

- **Detailed Metrics for Interconnection**  
  Extract circuit-level electrical results and metrics to quantitatively support DER hosting capacity and interconnection decisions.

- **GIS Circuit Visualization**  
  Integrate with `geopandas` and `folium` for interactive circuit mapping (compatible with shapefile-based network data).

- **Object-Oriented Extensible Architecture**  
  Easily customize and extend analysis pipelines for DER and network modeling needs.

## Why DERify?

OpenDSS (and thus its Python extensions) is the de facto open-source tool for distribution system simulation, widely used for DER impact studies. However, its built-in support for simulating dynamic (transient) faults involving inverter-interfaced DERs is limited:

- **Dynamic Fault Limitations for Inverter-Based DERs:**  
  As of 2023, OpenDSS primarily models inverter-based DERs (PV systems, BESS, etc.) for steady-state studies. During **dynamic fault analysis** (`SolveMode=Dynamic`), the OpenDSS solver **disables user-defined inverter models**, as there is no detailed internal inverter control or ride-through implementation in the time-domain solution (see [EPRI OpenDSS Manual, Section 12.18–12.22](https://sourceforge.net/projects/electricdss/files/OpenDSSManual/)).  
  > *"Note that elements such as PVSystem, Storage, and WindGen are turned OFF automatically when the Dynamic simulation starts. These components are steady‐state models and do not have dynamic models."*  
  > — OpenDSS Manual

- **Workaround Used in DERify:**  
  To evaluate worst-case network fault currents (e.g. for protection studies), DERify **temporarily replaces inverter DERs with equivalent voltage-source Generator elements**. These "dummy" generators mimic the capacity and topology of the DERs, ensuring correct short-circuit calculations. This approach, found in the `host.py` module, follows best practice workarounds in the literature and industry ([EPRI, 2016](https://ieeexplore.ieee.org/document/7553425); [Quint et al., 2022](https://ieeexplore.ieee.org/document/9692165)).

## Example Workflow

```python
from derify import DERCircuit

# Setup circuit with DER config
ckt = DERCircuit(
    ckt_path="./CKT/CKT_Daily.dss",
    bess_attrs=[{'storage_id': "mv_battery", 'bus_id': "busMV3", ...}],
    pvsys_attrs=[...]
)
losses = ckt.calculate_losses()
ckt.fault_network()
```

For detailed usage and options, see [docs/usage.md](docs/usage.md).

## Installation

DERify depends on:

- [dss-python](https://github.com/dss-extensions/dss_python) (`pip install dss-python`)
- `numpy`, `geopandas`, `folium`, `matplotlib`

Install dependencies via [requirements.txt](requirements.txt).

## Citation and References

- OpenDSS Manual: [sourceforge.net/projects/electricdss/files/OpenDSSManual/](https://sourceforge.net/projects/electricdss/files/OpenDSSManual/)
- Quint, R., et al. "Fault Current Contributions from Inverter-Based Resources: A Review." _IEEE Trans. Power Delivery_, 2022. [doi:10.1109/TPWRD.2022.3146750](https://ieeexplore.ieee.org/document/9692165)
- EPRI, "Distribution Fault Current Contributions from Distributed Energy Resources, 2016." [PDF](https://ieeexplore.ieee.org/document/7553425)

## License

Distributed under the MIT License. See [LICENSE](LICENSE).

## Acknowledgements

DERify utilizes the [dss-python](https://github.com/dss-extensions/dss_python) engine and builds on OpenDSS concepts pioneered by EPRI.

---

**Note**: For limitations and best practices regarding dynamic studies with inverters, see the [FAQ](docs/faq.md).
