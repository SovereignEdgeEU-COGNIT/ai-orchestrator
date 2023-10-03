import { ContentHeader } from '@components';
import React, { Component, useContext, useRef, useEffect } from "react"
import * as d3 from "d3";

const CirclePacking = ({ data, width, height, backgroundColor }) => {
    const svgRef = useRef(null);

    useEffect(() => {
        const svg = d3.select(svgRef.current);
        const pack = d3.pack()
            .size([width, height])
            .padding(12);

        const root = d3.hierarchy(data)
            .sum(d => d.value || 1)  // This ensures every node has a value.
            .sort((a, b) => b.value - a.value);

        const maxRadius = 60
        const nodes2 = pack(root).descendants();

        if (root.children && root.children.length === 1) {
            const singleChild = nodes2[1];
            singleChild.r = maxRadius;
            singleChild.x = width / 2;
            singleChild.y = height / 2;
        }

        svg.selectAll('rect').remove();
        svg.selectAll('*').remove();

        const numCols = Math.ceil(Math.sqrt(nodes2.length - 1));
        const numRows = Math.ceil((nodes2.length - 1) / numCols);
        const boxSize = Math.min(width / numCols, height / numRows);
        const padding = 5;

        const rect = svg.selectAll('rect')
            .data(nodes2)
            .enter().append('rect');

        rect
            .attr('rx', 5)
            .attr('ry', 5)
            .attr('x', (d, i) => {
                if (!d.parent) return 0;
                const col = (i - 1) % numCols;
                return col * (boxSize + padding);
            })
            .attr('y', (d, i) => {
                if (!d.parent) return 0;
                const row = Math.floor((i - 1) / numCols);
                return row * (boxSize + padding);
            })
            .attr('width', d => !d.parent ? width : boxSize)
            .attr('height', d => !d.parent ? height : boxSize)
            .style('fill', d => {
                if (!d.parent) return backgroundColor;
                if (!d.children) return "#BAC6BE";
                return '#f00';
            });

        const text = svg.selectAll('text')
            .data(nodes2)
            .enter().append('text');

        text
            .attr('x', (d, i) => {
                if (!d.parent) return width / 2;
                const col = (i - 1) % numCols;
                return col * (boxSize + padding) + boxSize / 2;
            })
            .attr('y', (d, i) => {
                if (!d.parent) return height / 2;
                const row = Math.floor((i - 1) / numCols);
                return row * (boxSize + padding) + boxSize / 2 + 5;
            })
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'middle')
            .text(d => d.parent ? d.data.name : '')
            .style('fill', '#333')
            .style('font-size', '12px');


        const nodeGroup = svg.selectAll('g.nodeGroup')
            .data(nodes2)
            .enter().append('g')
            .attr('class', 'nodeGroup')
            .attr('transform', d => `translate(${d.x},${d.y})`);
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
