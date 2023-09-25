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
            .sum(d => d.value)
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
                if (!d.children) return '#58a';
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

const DashboardView = (props) => {
    let stats = props.stats

    const host1 = {
        name: "root",
        color: "#0074D9",
        children: [
            { name: "VM: 1", value: 10 },
            // { name: "VM: 2", value: 50 },
            // { name: "VM: 6", value: 100 },
            // { name: "VM: 7", value: 100 },
            // { name: "VM: 8", value: 100 },
            // { name: "VM: 9", value: 100 },
        ]
    };

    const host2 = {
        name: "root",
        color: "#0074D9",
        children: [
            { name: "VM: 2", value: 70 },
            { name: "VM: 3", value: 70 },
        ]
    };

    const host3 = {
        name: "root",
        color: "#0074D9",
        children: [
            { name: "VM: 4", value: 70 },
            { name: "VM: 5", value: 70 },
            { name: "VM: 5", value: 70 },
            { name: "VM: 5", value: 70 },
            { name: "VM: 5", value: 70 },
            { name: "VM: 5", value: 70 },
            { name: "VM: 5", value: 70 },
            { name: "VM: 5", value: 70 },
        ]
    };

    return (
        <div>
            <ContentHeader title="Virtual Machine Placement" />
            <section className="content">
                <div className="container-fluid">
                    <div className="card">
                        <div className="card-body">
                            <table>
                                <tbody>
                                    <tr>
                                        <td>
                                            <h4 style={{ textAlign: 'center' }}>Host: 1</h4>
                                            <PlacementLayout backgroundColor="#20A300" data={host1} />
                                        </td>
                                        <td>
                                            <h4 style={{ textAlign: 'center' }}>Host: 2</h4>
                                            <PlacementLayout backgroundColor="#CA9D00" data={host2} />
                                        </td>
                                        <td>
                                            <h4 style={{ textAlign: 'center' }}>Host: 3</h4>
                                            <PlacementLayout backgroundColor="#20A300" data={host3} />
                                        </td>
                                    </tr>
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
            stats: {},
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
    }

    componentWillUnmount() {
        clearInterval(this.interval)
    }

    render() {
        const { stats } = this.state
        return (
            <div>
                <DashboardView stats={stats} />
            </div>
        );
    }
}

export default Page;
