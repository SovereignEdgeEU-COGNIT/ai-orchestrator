import { ContentHeader } from '@components';
import React, { Component, useContext, useRef, useEffect } from "react"
import * as d3 from "d3";

const CirclePacking = ({ data, width, height, backgroundColor }) => {
    const svgRef = useRef(null);

    useEffect(() => {

        const svg = d3.select(svgRef.current);

        const pack = d3.pack()
            .size([width, height])
            .padding(2);

        const root = d3.hierarchy(data)
            .sum(d => d.value || 1)  // This ensures every node has a value.
            .sort((a, b) => b.value - a.value);

        const maxRadius = 70
        const nodes2 = pack(root).descendants();

        if (root.children && root.children.length === 1) {
            const singleChild = nodes2[1];
            singleChild.r = maxRadius;
            singleChild.x = width / 2;
            singleChild.y = height / 2;
        }

        svg.selectAll('circle').remove();
        svg.selectAll('*').remove();

        const circle = svg.selectAll('circle')
            .data(nodes2)
            .enter().append('circle');

        circle
            .attr('r', d => d.r)
            .attr('cx', d => d.x)
            .attr('cy', d => d.y)
            .attr('text-anchor', 'middle')
            .text(d => {
                if (d.parent) {
                    return d.data.name;
                }
                return '';
            })
            .style('fill', d => {
                if (!d.parent) return backgroundColor;
                if (!d.children) return '#BAC6BE';
                if (d.parent === root) return '#f00';
                return '#eee';
            })

        const nodeGroup = svg.selectAll('g.nodeGroup')
            .data(nodes2)
            .enter().append('g')
            .attr('class', 'nodeGroup')
            .attr('transform', d => `translate(${d.x},${d.y})`);

        nodeGroup.append('text')
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'central')
            .text(d => {
                if (!d.parent) return '';
                return d.data.name || '';
            })
            .style('fill', '#333')
            .style('font-size', '17px');

    }, [data, width, height]);

    return (
        <svg ref={svgRef} width={width} height={height}></svg>
    );
};

const PlacementLayout = ({ backgroundColor, data }) => {
    return <CirclePacking data={data} width={300} height={300} backgroundColor={backgroundColor} />;
}

function chunkArray(myArray, chunk_size) {
    var results = [];

    while (myArray.length) {
        results.push(myArray.splice(0, chunk_size));
    }

    return results;
}

function transform(jsonData) {
    return jsonData.map(item => {
        return {
            name: `Host: ${item.hostid}`,
            children: item.vmids.map(vmid => ({
                name: `VM: ${vmid}`,
                value: 70
            }))
        };
    });
}

const DashboardView = (props) => {
    let placementLayout = props.stats
    let data = transform(placementLayout)

    let items = [];
    for (let i = 0; i < placementLayout.length; i++) {
        if (placementLayout[i].state.renewable_energy) {
            items.push(
                <td>
                    <h4 style={{ textAlign: 'center' }}>Host: {placementLayout[i].hostid}</h4>
                    <PlacementLayout backgroundColor="#68947B" data={data[i]} />
                </td>
            );
        } else {
            items.push(
                <td>
                    <h4 style={{ textAlign: 'center' }}>Host: {placementLayout[i].hostid}</h4>
                    <PlacementLayout backgroundColor="#BE947B" data={data[i]} />
                </td>
            );
        }
    }
    let rows = chunkArray(items, 3);

    return (
        <div>
            <ContentHeader title="Virtual Machine Placement" />
            <section className="content">
                <div className="container-fluid">
                    <div className="card">
                        <div className="card-body">
                            <table>
                                <tbody>
                                    {rows.map((row, rowIndex) => (
                                        <tr key={rowIndex}>
                                            {row.map((cell, cellIndex) => (
                                                <td key={cellIndex} style={{ padding: '15px' }}>
                                                    {cell}
                                                </td>
                                            ))}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </section>

        </div>
    );
};

class Page extends Component {
    constructor() {
        super();
        this.state = {
            placementLayout: [],
            show: false
        };
    }

    setShow = (show) => {
        this.setState({ show });
    };

    setMessage = (message) => {
        this.setState({ message });
    };

    setHeading = (heading) => {
        this.setState({ heading });
    };

    componentDidMount() {
        fetch('http://localhost:8000/') // Replace with your specific endpoint if needed
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(rawPlacementLayout => {
                let placementLayout = rawPlacementLayout.sort((a, b) => {
                    return Number(a.hostid) - Number(b.hostid);
                })

                placementLayout.forEach(host => {
                    host.vmids.sort((a, b) => Number(a) - Number(b));
                });

                this.setState({ placementLayout });
            })
            .catch(error => {
                // this.setState({ error, isLoading: false });
            });

        this.interval = setInterval(() => {
            fetch('http://rocinante:8000/') // Replace with your specific endpoint if needed
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(rawPlacementLayout => {
                    let placementLayout = rawPlacementLayout.sort((a, b) => {
                        return Number(a.hostid) - Number(b.hostid);
                    })

                    placementLayout.forEach(host => {
                        host.vmids.sort((a, b) => Number(a) - Number(b));
                    });

                    this.setState({ placementLayout });
                })
                .catch(error => {
                    // this.setState({ error, isLoading: false });
                });
        }, 1000)
    }

    componentWillUnmount() {
        clearInterval(this.interval)
    }

    render() {
        const { placementLayout } = this.state
        return (
            <div>
                <DashboardView stats={placementLayout} />
            </div>
        );
    }
}

export default Page;
