import 'package:flutter/material.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: Text('VoicePasskey Mobile')),
        body: Center(child: Text('Flutter stub - Integrate JS logic via webview')),
      ),
    );
  }
}
