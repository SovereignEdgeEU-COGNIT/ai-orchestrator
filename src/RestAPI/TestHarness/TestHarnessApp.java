package TestHarness;

import javax.swing.*;
import java.awt.BorderLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

public class TestHarnessApp
{

    final static String versionServiceEndpoint = "http://127.0.0.1:8000/version";
    final static String monitorServiceEndpoint = "http://127.0.0.1:8000/cognitMonitor";

    static JFrame theMainWindow;
    static JPanel bottomPanel;
    static JButton quickTestButton;
    static JButton serviceVersionButton;
    static JButton submitFileButton;
    static JLabel outputLabel;

    public static void main(String args[])
    {
        theMainWindow = new JFrame();
        theMainWindow.setTitle("COGNIT WP4 Test Harness");
        theMainWindow.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        theMainWindow.setSize(800,800);
        
        outputLabel = new JLabel("No output yet received");
        outputLabel.setHorizontalAlignment(JLabel.CENTER);

        serviceVersionButton = new JButton("Check service version");
        serviceVersionButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                checkServiceVersion();
            }
        });


        quickTestButton = new JButton("Quick monitor test");
        quickTestButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                performQuickTest();
            }
        });

        submitFileButton = new JButton("Upload a JSON file to test");
        
        bottomPanel = new JPanel();
        bottomPanel.add(serviceVersionButton);
        bottomPanel.add(quickTestButton);
        bottomPanel.add(submitFileButton);        
        
        theMainWindow.add(outputLabel, BorderLayout.CENTER);
        theMainWindow.add(bottomPanel, BorderLayout.SOUTH);

        theMainWindow.setVisible(true);
    }

    private static void performQuickTest()
    {
        outputLabel.setText("Boo");       
    }

    private static void checkServiceVersion()
    {
        try
        {
            URI theService = new URI(versionServiceEndpoint);
            HttpRequest theRequest = HttpRequest.newBuilder()
                .uri(theService)
                .GET()
                .build();

            HttpClient theClient = HttpClient.newHttpClient();
            HttpResponse<String> theResponse = theClient.send(theRequest, HttpResponse.BodyHandlers.ofString());
            
            outputLabel.setText("<html>" + (String) theResponse.body() + "</html>");  // put it in html tags to make it wrap on the screen
        }

        catch (Exception e)
        {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            outputLabel.setText("<html>Exception:  " + sw.toString() + "</html>");  // put it in html tags to make it wrap on the screen   
        }
    }
}